import os
import torch
import torch.nn.functional as F
from PIL import Image

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from geochat.conversation import conv_templates, SeparatorStyle
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import tokenizer_image_token, get_model_name_from_path


class VQAModel:
    def __init__(self, model_path: str, model_base: str | None = None, conv_mode: str = "llava_v1", load_4bit: bool = True):
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, model_name, load_4bit=load_4bit,
        )
        self.conv_mode = conv_mode

    def answer(self, image_path: str, question: str) -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"image not found: {image_path}")

        conv = conv_templates[self.conv_mode].copy()
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_processor.preprocess(
            [image],
            crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504},
            return_tensors="pt",
        )["pixel_values"].half().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.0,
                num_beams=1,
                max_new_tokens=256,
                length_penalty=2.0,
                use_cache=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        input_token_len = input_ids.shape[1]
        generated_ids = out.sequences[:, input_token_len:]
        text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()

        confidence = self._mean_token_confidence(out.scores, generated_ids[0])
        return {"answer": text, "confidence": confidence}

    @staticmethod
    def _mean_token_confidence(scores, generated_ids) -> float:
        if len(scores) == 0:
            return 0.0
        probs = []
        for step_logits, token_id in zip(scores, generated_ids):
            step_probs = F.softmax(step_logits[0].float(), dim=-1)
            probs.append(step_probs[token_id].item())
        return sum(probs) / len(probs) if probs else 0.0
