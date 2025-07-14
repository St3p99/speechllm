import logging

from speechllm.constants import DEFAULT_AUDIO_TOKEN_IDX, IGNORE_INDEX
from speechllm.conversation import Conversation
from speechllm.model.configuration_speechllm import SpeechLLMConfig
from speechllm.model.projector import MLPProjector
from speechllm.model.speech_encoder import WavLMEncoder
from speechllm.model.text_decoder import LlamaDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SpeechLLM(nn.Module):
    def __init__(
        self,
        config: SpeechLLMConfig,
    ):
        super().__init__()
        self.config = config
        self.log_eval = True

        if "wavlm" in config.speech_encoder_name_or_path.lower():
            self.encoder = WavLMEncoder(
                _name_or_path=config.speech_encoder_name_or_path,
                stack_size=config.downsample_factor,
            )
        else:
            raise ValueError(
                f"Unsupported speech encoder: {config.speech_encoder_name_or_path}"
            )

        if "llama-3" in config.text_decoder_name_or_path.lower():
            self.text_decoder = LlamaDecoder(
                name_or_path=config.text_decoder_name_or_path,
                conversation_version=config.conversation_version,
                config_dict={"tie_word_embeddings": config.tie_word_embeddings},
            )
        else:
            raise ValueError(
                f"Unsupported text decoder: {config.text_decoder_name_or_path}"
            )

        self.projector = MLPProjector(
            input_dim=self.encoder.output_hidden_size,
            output_dim=self.text_decoder.hidden_size,
            hidden_size=self.config.projector_hidden_size,
            hidden_layers=self.config.projector_n_layers,
            residual=True,
            activation=self.config.projector_activation,
        )

        # set trainable parameters
        self._set_trainable_parameters()

    def _set_trainable_parameters(self):
        self.requires_grad_(True)
        for module_name in self.config.freeze_modules:
            module = getattr(self, module_name)
            if module is not None:
                module.requires_grad_(False)

        all_trainable_params = 0

        # Report which modules and how many parameters are trainable
        for name, module in self.named_children():
            is_trainable = any(p.requires_grad for p in module.parameters())
            status = "✅ TRAINABLE" if is_trainable else "❄️ FROZEN"
            param_count = sum(p.numel() for p in module.parameters())
            print(f"  {name}: {status} ({param_count:,} params)")
            # Print trainable submodules if any
            if is_trainable:
                print(f"    Trainable submodules in {name}:")
                for sub_name, sub_module in module.named_modules():
                    if sub_name == "":
                        continue
                    trainable_params = [
                        p
                        for p in sub_module.parameters(recurse=False)
                        if p.requires_grad
                    ]
                    if trainable_params:
                        sub_param_count = sum(p.numel() for p in trainable_params)
                        all_trainable_params += sub_param_count
                        print(f"      {sub_name}: {sub_param_count:,} trainable params")

        # Count total parameters
        total_params = sum(p.numel() for p in self.parameters())

        # Validate that at least some parameters are trainable
        if all_trainable_params == 0:
            raise RuntimeError(
                "No trainable parameters found! The model will not learn. "
                "Check your freeze_modules setting."
            )

        percent_trainable = (
            (100 * all_trainable_params / total_params) if total_params > 0 else 0.0
        )
        print(
            f"🔧 Trainable parameters: {all_trainable_params:,} / {total_params:,} "
            f"({percent_trainable:.1f}%)"
        )

    def forward(
        self,
        input_ids,
        audios=None,
        audio_padding_masks=None,
        attention_mask=None,
        position_ids=None,
        input_embeds=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if input_embeds is None:
            # build inputs_embeds, labels,.. from text and audio inputs
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                audio_features,
            ) = self.prepare_inputs_labels(
                text_inputs={
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "labels": labels,
                },
                audio_inputs={
                    "audios": audios,
                    "audio_padding_masks": audio_padding_masks,
                },
            )
        else:
            inputs_embeds = input_embeds

        model_output = self.text_decoder(
            input_ids=input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return model_output

    def prepare_inputs_labels(self, text_inputs, audio_inputs):
        """
        Prepare inputs and labels from text and audio inputs
        Args:
            text_inputs: dict with keys "input_ids", "position_ids", "attention_mask", "past_key_values", "labels"
            audio_inputs: dict with keys "audios_srs"
        Returns:
            tuple: (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, audio_features)
        """

        input_ids = text_inputs.get("input_ids")
        position_ids = text_inputs.get("position_ids")
        attention_mask = text_inputs.get("attention_mask")
        past_key_values = text_inputs.get("past_key_values", None)
        labels = text_inputs.get("labels", None)

        # assert that labels does not contain only ignore index
        if labels is not None and (labels == IGNORE_INDEX).all():
            raise ValueError("Labels contain only ignore index")

        audios = audio_inputs.get("audios", None)
        audio_padding_masks = audio_inputs.get("audio_padding_masks", None)

        if audios is None:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,  # inputs_embeds
                labels,
                None,  # audio_features
            )

        # Process batch of audios
        audio_valid_mask = (
            ~audio_padding_masks if audio_padding_masks is not None else None
        )
        if audios is not None:
            # Pass batched tensor and attention masks to encoder
            audio_features, audio_attention_mask = self.encoder(
                audios,
                attention_mask=audio_valid_mask,
                return_attention_mask=True,
            )
            projector_output = self.projector(audio_features)
            audio_features = {
                "audio_features": audio_features,
                "projected_audio_features": projector_output,
                "audio_attention_mask": audio_attention_mask,
            }

        # Combine text and audio inputs
        # ------------------------------------------------

        # Handle None values by creating default tensors
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device,
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove padding using attention_mask
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        # Initialize lists for processed inputs and labels
        new_input_embeds = []
        new_labels = []

        for (
            batch_idx,
            cur_input_ids,
        ) in enumerate(input_ids):
            cur_labels = labels[batch_idx]

            cur_new_input_embeds, cur_new_labels = self.insert_multimodal_features(
                cur_input_ids, cur_labels, audio_features, batch_idx
            )

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as audio embeddings
        model_max_length = self.text_decoder.tokenizer.model_max_length
        max_len = max(x.shape[0] for x in new_input_embeds)
        if model_max_length is not None and max_len > model_max_length:
            logger.info(f"Truncating sequences to model max length: {model_max_length}")
            max_len = model_max_length
            new_input_embeds = [x[:model_max_length] for x in new_input_embeds]
            new_labels = [x[:model_max_length] for x in new_labels]

        # Stack them back as a single tensor, padding if necessary
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )

        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        # Pad the embeddings and labels
        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            padding_side = getattr(self.text_decoder.tokenizer, "padding_side", "right")
            if padding_side == "left":
                logger.warning(f"Padding left for batch {i}")
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:
                # Right padding
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        inputs_embeds = torch.stack(new_input_embeds_padded, dim=0)
        attention_mask = attention_mask
        position_ids = position_ids
        labels = new_labels_padded

        # DEBUG: check that the inputs_embeds are correct
        # for i in range(len(inputs_embeds)):
        #     p_idx = torch.where(input_ids[i] == DEFAULT_AUDIO_TOKEN_IDX)[0][0].item()
        #     text_embeds = self.text_decoder.model.get_input_embeddings()(input_ids[i][:p_idx])
        #     assert torch.allclose(text_embeds, inputs_embeds[i][:p_idx])

        #     text_embeds = self.text_decoder.model.get_input_embeddings()(input_ids[i][p_idx+1:])
        #     audio_len = projector_output[i][audio_attention_mask[i]].shape[0]
        #     input_embeds_after_audio = inputs_embeds[i][p_idx+audio_len:][attention_mask[i][p_idx+audio_len:]]
        #     assert torch.allclose(text_embeds, input_embeds_after_audio)

        #     audio_embeds = projector_output[i][audio_attention_mask[i]]
        #     assert torch.allclose(audio_embeds.to(torch.bfloat16), inputs_embeds[i][p_idx:p_idx+audio_len].to(torch.bfloat16))

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            audio_features,
        )

    def insert_multimodal_features(self, input_ids, labels, audio_features, batch_idx):
        """
        Insert multimodal features (e.g., audio) into the text embeddings and labels at the appropriate positions.
        Args:
            input_ids: torch.Tensor, shape (n_text_tokens,)
            labels: torch.Tensor, shape (n_text_tokens,)
            audio_features: dict with keys "audio_features", "projected_audio_features", "audio_attention_mask"
            batch_idx: int, index of the current batch item
        Returns:
            new_input_embeds: torch.Tensor, shape (n_total_tokens, hidden_size)
            new_labels: torch.Tensor, shape (n_total_tokens,)
        """
        # Find all audio placeholder indices
        audio_placeholder_indices = (
            (input_ids == DEFAULT_AUDIO_TOKEN_IDX).nonzero(as_tuple=True)[0].tolist()
        )
        assert (
            len(audio_placeholder_indices) == 1
        ), "Only one audio placeholder token is supported"

        new_input_embeds = []
        new_labels = []

        last_idx = 0

        for idx in audio_placeholder_indices:
            # Add text tokens before the audio placeholder
            if idx > last_idx:
                text_embeds = self.text_decoder.model.get_input_embeddings()(
                    input_ids[last_idx:idx]
                )
                new_input_embeds.append(
                    text_embeds
                )  # Use the entire text_embeds tensor
                new_labels.append(labels[last_idx:idx])

            # Insert audio features for this batch
            if audio_features is not None:
                projected_audio_features = audio_features["projected_audio_features"][
                    batch_idx
                ]
                audio_attention_mask = audio_features["audio_attention_mask"][batch_idx]
                audio_to_append = projected_audio_features[audio_attention_mask].to(
                    device=input_ids.device
                )
                new_input_embeds.append(audio_to_append)
                new_labels.append(
                    torch.full(
                        (audio_to_append.shape[0],),
                        fill_value=IGNORE_INDEX,
                        device=input_ids.device,
                        dtype=labels.dtype,
                    )
                )

            last_idx = idx + 1  # Skip the placeholder token

        # Add any remaining text tokens after the last audio segment
        if last_idx < input_ids.shape[0]:
            text_embeds = self.text_decoder.model.get_input_embeddings()(
                input_ids[last_idx:]
            )
            new_input_embeds.append(text_embeds)
            new_labels.append(labels[last_idx:])

        # Concatenate all parts
        new_input_embeds = torch.cat(new_input_embeds, dim=0)
        new_labels = torch.cat(new_labels, dim=0)

        return new_input_embeds, new_labels

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self):
        """Get the dtype of the model."""
        return next(self.parameters()).dtype


if __name__ == "__main__":
    pass
