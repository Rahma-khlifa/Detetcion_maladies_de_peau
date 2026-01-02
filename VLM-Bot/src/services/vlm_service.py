"""
VLM Service - LLaVA-1.5-7B Vision Language Model
Gère le chargement et l'inférence du modèle VLM local.
Optimisé pour GPU 6GB VRAM avec quantization 4-bit.
"""

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from typing import Optional, Dict, Any
from PIL import Image
import logging
from pathlib import Path
import gc

logger = logging.getLogger(__name__)


class VLMService:
    """Service pour le modèle Vision-Language LLaVA."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le service VLM.
        
        Args:
            config: Configuration du modèle (dict depuis config.yaml)
        """
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self) -> None:
        """
        Charge le modèle VLM avec quantization 4-bit.
        
        LLaVA-1.5-7B - optimisé pour 6GB VRAM.
        """
        model_name = self.config['name']
        
        logger.info(f"🔄 Chargement du modèle VLM: {model_name}")
        
        # Nettoyer la mémoire GPU avant chargement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Configuration de quantization 4-bit
        quant_config = BitsAndBytesConfig(
            load_in_4bit=self.config['quantization']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(
                torch, 
                self.config['quantization']['bnb_4bit_compute_dtype']
            ),
            bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type']
        )
        
        # Préparer max_memory
        max_memory = {}
        if isinstance(self.config.get('max_memory'), dict):
            for k, v in self.config['max_memory'].items():
                # Convertir "5GB" → "5GiB"
                if isinstance(v, str):
                    max_memory[int(k) if k.isdigit() else k] = v.replace("GB", "GiB")
                else:
                    max_memory[k] = v
        
        try:
            logger.info("📦 Chargement avec quantization 4-bit...")
            
            # Charger le modèle LLaVA
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=self.config.get('device_map', 'auto'),
                max_memory=max_memory if max_memory else None,
                torch_dtype=getattr(torch, self.config.get('torch_dtype', 'float16')),
                low_cpu_mem_usage=True
            )
            
            logger.info("✅ Modèle LLaVA chargé avec succès!")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
        
        # Charger le processor
        logger.info("📦 Chargement du processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Afficher info mémoire
        self._log_memory_info()
    
    def _cleanup(self):
        """Nettoie mémoire entre tentatives."""
        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _log_memory_info(self):
        """Affiche les informations de mémoire."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"✅ Modèle VLM prêt!")
            logger.info(f"   GPU allocated: {allocated:.2f} GB")
            logger.info(f"   GPU reserved: {reserved:.2f} GB")
        else:
            logger.info(f"✅ Modèle VLM prêt (mode CPU)")
        
        # Afficher device_map
        if hasattr(self.model, 'hf_device_map'):
            devices = set(str(v) for v in self.model.hf_device_map.values())
            logger.info(f"   Devices utilisés: {devices}")
            
            # Compter layers sur chaque device
            device_counts = {}
            for layer, device in self.model.hf_device_map.items():
                device_counts[str(device)] = device_counts.get(str(device), 0) + 1
            
            for device, count in device_counts.items():
                logger.info(f"   {device}: {count} layers")
    
    def generate_diagnosis(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.5,
        do_sample: bool = True
    ) -> str:
        """
        Génère un diagnostic basé sur l'image et le prompt.
        
        Args:
            image: Image PIL à analyser
            prompt: Question/instruction pour le VLM
            max_new_tokens: Nombre max de tokens à générer
            temperature: Contrôle de la randomness (0.0-1.0)
            do_sample: Utiliser sampling ou greedy decoding
            
        Returns:
            Diagnostic généré par le VLM
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Modèle non chargé. Appelez load_model() d'abord.")
        
        try:
            # Nettoyer cache GPU avant génération
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Format du prompt pour LLaVA
            # Format: USER: <image>\n{prompt}\nASSISTANT:
            formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
            
            # Traiter l'image et le texte
            inputs = self.processor(
                text=formatted_prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Déplacer inputs vers GPU avec le bon dtype
            model_device = self.device if self.device == "cuda" else "cpu"
            inputs = {
                k: v.to(model_device, torch.float16) if hasattr(v, 'to') and v.dtype == torch.float32 
                else v.to(model_device) if hasattr(v, 'to') 
                else v 
                for k, v in inputs.items()
            }
            
            # Générer la réponse
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.9 if do_sample else None
                )
            
            # Décoder la sortie complète
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extraire seulement la réponse (après ASSISTANT:)
            if "ASSISTANT:" in generated_text:
                generated_text = generated_text.split("ASSISTANT:")[-1].strip()
            
            return generated_text
            
        except torch.cuda.OutOfMemoryError:
            logger.error("❌ GPU Out of Memory!")
            torch.cuda.empty_cache()
            return "Erreur: Mémoire GPU insuffisante. Réduisez max_new_tokens."
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération: {e}", exc_info=True)
            return f"Erreur: {str(e)}"
    
    def unload_model(self) -> None:
        """Libère la mémoire GPU."""
        self._cleanup()
        if self.processor is not None:
            del self.processor
            self.processor = None
        logger.info("✅ Modèle déchargé, mémoire libérée")
