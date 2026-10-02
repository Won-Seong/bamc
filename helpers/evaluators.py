"""
This file contains the evaluator classes for evaluating the quality of generated images.
"""

from typing import List, Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from insightface.app import FaceAnalysis

from helpers.llm import call_vlm
from helpers.models import VLMBatchDistanceResponse, VLMDistanceResponse

class Evaluator:
    """
    Base class for all evaluators.
    """
    def __init__(
        self,
        device=None
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.target = None
        self.name = None

    @torch.no_grad()
    def set_target(self, target: torch.Tensor):
        """This is for caching the target image."""
        raise NotImplementedError

    @torch.no_grad()
    def calculate_distance(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor
    ) -> float:
        """This is for calculating the distance between two images."""
        raise NotImplementedError

    @torch.no_grad()
    def calculate_distances_batch(
        self,
        imgs: List[torch.Tensor]
    ) -> List[float]:
        """This is for calculating the distance between two images in batch."""
        raise NotImplementedError

class LPIPSEvaluator(Evaluator):
    """
    Highly Optimized Learned Perceptual Image Patch Similarity (LPIPS) evaluator.
    """
    def __init__(self, device=None):
        super().__init__(device)
        self.model = LearnedPerceptualImagePatchSimilarity(
            net_type='vgg',
            reduction='none'
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.name = "LPIPS"
        self.to_tensor = transforms.ToTensor()

    def _prepare_input(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Safely converts PIL or Tensors to the strictly required [-1, 1] range 
        with a batch dimension [1, C, H, W].
        """
        if isinstance(img, Image.Image):
            # Convert PIL to Tensor [0, 1], add batch dim, scale to [-1, 1]
            img_rgb = img.convert('RGB')
            tensor = self.to_tensor(img_rgb).unsqueeze(0).to(self.device)
            return tensor * 2.0 - 1.0
            
        elif isinstance(img, torch.Tensor):
            tensor = img.to(self.device)
            
            # 1. Ensure it has a batch dimension (Shape: [B, C, H, W])
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)

            if tensor.shape[1] == 4:
                tensor = tensor[:, :3, :, :]
                
            # 2. Safety Check: If the tensor is in [0, 1] range, force it to [-1, 1]
            # (We check if the minimum value is >= 0, implying it hasn't been shifted yet)
            if tensor.max() <= 1.01 and tensor.min() >= -0.01:
                if tensor.min() >= 0.0:
                    tensor = tensor * 2.0 - 1.0
                
            return tensor
            
        else:
            raise TypeError(f"Expected PIL Image or Tensor, got {type(img)}")

    @torch.no_grad()
    def set_target(self, target: Union[Image.Image, torch.Tensor]):
        """Set the target image for the evaluator."""
        self.target = self._prepare_input(target)

    @torch.no_grad()
    def calculate_distance(self, img1: Union[Image.Image, torch.Tensor], img2: Union[Image.Image, torch.Tensor]) -> float:
        """Calculate the LPIPS distance between two images."""
        img1_t = self._prepare_input(img1)
        img2_t = self._prepare_input(img2)
        
        # Ensure img2_t matches the batch size of img1_t if needed
        if img1_t.shape[0] != img2_t.shape[0]:
            img2_t = img2_t.expand(img1_t.shape[0], -1, -1, -1)
            
        return self.model(img1_t, img2_t).item()

    @torch.no_grad()
    def calculate_distances_batch(self, imgs: List[Union[Image.Image, torch.Tensor]]) -> List[float]:
        """
        Calculate the LPIPS distance using highly optimized GPU vectorization.
        """
        if not imgs:
            return []

        # 1. Prepare all images (now guaranteed to be [1, 3, H, W] and in [-1, 1])
        prepared_imgs = [self._prepare_input(img) for img in imgs]

        # 2. Stack them into a single massive batch tensor: [B, 3, H, W]
        batch_tensor = torch.cat(prepared_imgs, dim=0)

        # 3. Duplicate the target image so it matches the batch size
        batch_size = batch_tensor.shape[0]
        target_batch = self.target.expand(batch_size, -1, -1, -1)

        # 4. Push the whole batch through the VGG network in ONE operation!
        distances = self.model(batch_tensor, target_batch)

        # 5. Flatten the result back into a standard Python list
        return distances.view(-1).tolist()

class ArcFaceEvaluator(Evaluator):
    """
    True ArcFace evaluator using the official InsightFace library.
    Uses SCRFD for detection and ArcFace (ResNet) for embeddings.
    """
    def __init__(self, device=None):
        super().__init__(device)
        # ctx_id: 0 means first GPU, -1 means CPU.
        self.ctx_id = 0 if device == "cuda" else -1
        
        # 'buffalo_l' is the standard, high-accuracy InsightFace model pack.
        # It will automatically download on the first run (~300MB).
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=self.ctx_id, det_size=(512, 512))
        self.app.det_model.det_thresh = 0.3
        
        self.target_embedding = None
        self.name = "ArcFace"

    def _extract_embedding(self, img: Image.Image) -> torch.Tensor:
        """Helper to extract the largest face's embedding from a PIL Image."""
        img = img.convert('RGB')

        # Convert PIL RGB to OpenCV BGR
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Detect faces and extract features
        faces = self.app.get(img_bgr)

        if not faces:
            return None

        # If multiple faces are found, sort by bounding box area and take the largest
        if len(faces) > 1:
            faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)

        # InsightFace provides 'normed_embedding' which is perfect for cosine similarity
        emb = torch.tensor(faces[0].normed_embedding)
        return emb

    @torch.no_grad()
    def set_target(self, target: Image.Image):
        """Set the target image for the evaluator."""
        emb = self._extract_embedding(target)
        if emb is None:
            raise ValueError("Target image does not contain a face.")

        # Add batch dimension [1, 512]
        self.target_embedding = emb.unsqueeze(0)

    @torch.no_grad()
    def calculate_distance(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate the ArcFace distance (1 - Cosine Similarity) between two images."""
        emb1 = self._extract_embedding(img1)
        emb2 = self._extract_embedding(img2)

        if emb1 is None or emb2 is None:
            raise ValueError("Images do not contain faces.")

        # Calculate cosine similarity and convert to distance
        cos_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        return 1.0 - cos_sim

    @torch.no_grad()
    def calculate_distances_batch(self, imgs: List[Image.Image]) -> List[float]:
        """
        Compare a list of images against the set target.
        Returns a distance of 1.0 if no face is found in a specific image.
        """
        if self.target_embedding is None:
            raise RuntimeError("You must call set_target() first.")

        distances = []
        for img in imgs:
            emb = self._extract_embedding(img)

            if emb is None:
                # Fallback penalty if face detection fails
                distances.append(1.0)
            else:
                cos_sim = F.cosine_similarity(self.target_embedding, emb.unsqueeze(0)).item()
                distances.append(1.0 - cos_sim)

        return distances

# ==========================
# VLM Evaluator
# ==========================

class VLMEvaluator(Evaluator):
    """
    VLM evaluator.

    VLM evaulator is slightly different from other evaluators.
    calculate_distance is not implemented.
    calculate_distances_batch requires different parameters.
    """
    def __init__(
        self,
        client,
        prompt: List[str],
        model:str="gemini-3.1-flash-lite",
        device=None
    ):
        """
        Initialize the VLMEvaluator.
        
        Args:
            client: The Gemini client.
            prompt: The prompt for the VLM.
            model: The VLM model to use.
            device: The device to use.
        """
        super().__init__(device)
        self.client = client
        self.model = model
        self.name = "VLM"
        self.target = None
        self.prompt = prompt

    @torch.no_grad()
    def set_target(self, target: Image.Image):
        self.target = target

    @torch.no_grad()
    def calculate_distance(
        self,
        img: Image.Image
    ) -> Tuple[str, float]:
        result = call_vlm(
            client=self.client,
            imgs= [self.target, img],
            model=self.model,
            prompt=self.prompt[0],
            temperature=0.0,
            response_format=VLMDistanceResponse
        )
        result = VLMDistanceResponse.model_validate_json(result)
        return result.reasoning, 1 - (result.score / 100)

    @torch.no_grad()
    def calculate_distances_batch(
        self,
        existing_img: Image.Image,
        new_img: Image.Image
    ) -> Tuple[str, float]:
        """
        This NOT calculate distances. It outputs just VLM's score.
        VLM assess two images to the target image.

        Args:
            existing_img: Existing image.
            new_img: New image.

        Returns:
            float: The distance between the existing image and the new image.
                   If new_img is better than existing_img, return positive value.
                   else return negative value.
                   The score is in the range of 0 ~ 1.
        """
        result = call_vlm(
            client=self.client,
            imgs= [self.target, existing_img, new_img],
            model=self.model,
            prompt=self.prompt[1],
            temperature=0.0,
            response_format=VLMBatchDistanceResponse
        )
        result = VLMBatchDistanceResponse.model_validate_json(result)
        # Makes score based on the best_index and score,
        # If the best_index is 1, return [distance, 100]
        # else return [100, distance]
        # It makes the process determines the best image.
        selection = result.selection
        score = 1 - (result.score / 100) # 0 ~ 1

        if selection == "existing":
            return result.reasoning, score * -1 # -1 ~ 0
        else:
            return result.reasoning, score
