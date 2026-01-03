"""
Application Gradio - VLM-Bot
Interface web pour l'analyse dermatologique.
"""

import gradio as gr
from PIL import Image
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from services.vlm_service import VLMService
from services.rag_service import RAGService
from utils.helpers import (
    load_config,
    setup_logging,
    load_environment,
    ensure_directories,
    format_prompt
)

# Setup
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Charger configuration et environnement
try:
    load_environment()
    config = load_config()
    ensure_directories(config)
except Exception as e:
    logger.error(f"❌ Erreur de configuration: {e}")
    raise

# Initialiser les services globaux
vlm_service = None
rag_service = None


def initialize_services():
    """Initialise les services VLM et RAG (lazy loading)."""
    global vlm_service, rag_service
    
    if vlm_service is None:
        logger.info("🔄 Initialisation du VLM...")
        vlm_service = VLMService(config['models']['vlm'])
        vlm_service.load_model()
    
    if rag_service is None:
        logger.info("🔄 Chargement de l'index RAG...")
        rag_service = RAGService(config['rag'])
        index_path = config['rag'].get('index_path', 'data/processed/faiss_index')
        
        if Path(index_path).exists():
            rag_service.load_index(index_path)
        else:
            logger.warning("⚠️  Index RAG non trouvé. Construction en cours...")
            rag_service.build_index(save_path=index_path)


def analyze_lesion_complete(
    image,
    additional_context,
    max_tokens,
    temperature,
    num_sources
):
    """
    Pipeline complet d'analyse.
    
    Args:
        image: Image PIL
        additional_context: Contexte additionnel optionnel
        max_tokens: Max tokens à générer
        temperature: Température de sampling
        num_sources: Nombre de sources RAG
        
    Returns:
        Tuple (sources_text, diagnosis_text)
    """
    if image is None:
        return "", "⚠️ Veuillez télécharger une image d'abord!"
    
    try:
        # Initialiser les services (lazy)
        initialize_services()
        
        # Convertir image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        
        # Phase 1: Recherche RAG
        logger.info("📚 Recherche RAG...")
        key_terms = [
            "melanoma", "atypical nevus", "dysplastic nevus",
            "asymmetry", "irregular borders", "multiple colors",
            "pigmented lesion", "ABCDE criteria", "basal cell carcinoma",
            "squamous cell carcinoma", "dermatoscopy", "skin cancer"
        ]
        query_text = " ".join(key_terms)
        rag_results = rag_service.search(query_text, top_k=int(num_sources))
        
        # Formater les sources
        sources_text = f"**Found {len(rag_results)} relevant medical abstracts:**\n\n"
        
        retrieved_context = ""
        for i, (doc, score) in enumerate(rag_results, 1):
            sources_text += f"**[Source {i}]** (Relevance: {score:.4f})\n"
            sources_text += f"{doc.page_content}\n"
            sources_text += f"{'-'*80}\n\n"
            retrieved_context += f"\n[Source {i}]:\n{doc.page_content}\n"
        
        # Phase 2: Construire le prompt
        if additional_context and additional_context.strip():
            prompt = format_prompt(additional_context, retrieved_context, "with_context")
        else:
            prompt = format_prompt("", retrieved_context, "direct")
        
        # Phase 3: Génération VLM
        logger.info("🤖 Génération du diagnostic...")
        diagnosis = vlm_service.generate_diagnosis(
            image=image,
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        
        # Sauvegarder le rapport
        report = f"""
SKIN LESION ANALYSIS REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Method: VLM (Vision Language Model) + RAG

EVIDENCE-BASED DIAGNOSIS:
{diagnosis}

RETRIEVED SOURCES:
{retrieved_context}

DISCLAIMER: For research and educational purposes only. NOT a substitute for
professional medical advice, diagnosis, or treatment. Consult a qualified dermatologist.
{'='*80}
"""

        filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Analyse terminée. Rapport: {filename}")
        
        # Retourner les résultats
        return sources_text, diagnosis
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n{traceback.format_exc()}"
        return "", error_msg


# ============================================================================
# Interface Gradio
# ============================================================================

custom_css = """
    .scrollable-output textarea {
        max-height: 500px !important;
        overflow-y: auto !important;
    }
    .gradio-container {
        max-width: 1400px !important;
    }
    #sources_output, #diagnosis_output {
        max-height: 500px;
        overflow-y: auto;
    }
"""

with gr.Blocks(
    title="VLM-Bot - Dermatological Analysis",
) as demo:
    
    gr.Markdown("""
    # 🔬 VLM-Bot - Système d'Analyse Dermatologique
    
    **Llava + RAG**
    
    - 🤖 **VLM**: Llava-1.5-7B avec quantisation 4-bit
    - 📚 **RAG**: Diagnostic basé sur la littérature médicale
    - 👁️ **Analyse**: Vision + Language pour diagnostic complet
    
    ⚠️ **DISCLAIMER**: Usage éducatif uniquement. Consultez toujours un dermatologue.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Étape 1: Upload Image")
            image_input = gr.Image(type="pil", label="Image de la lésion")
            
            gr.Markdown("### � Étape 2: Contexte Additionnel (Optionnel)")
            additional_context = gr.Textbox(
                label="Contexte clinique",
                placeholder="Ex: Patient de 45 ans, lésion évoluant depuis 6 mois...",
                lines=4,
                info="Informations supplémentaires pour l'analyse"
            )
            
            gr.Markdown("### ⚙️ Étape 3: Paramètres")
            with gr.Accordion("Paramètres avancés", open=False):
                max_tokens = gr.Slider(
                    512, 2048, value=1024, step=128,
                    label="Tokens de génération"
                )
                temperature = gr.Slider(
                    0.1, 1.0, value=0.5, step=0.1,
                    label="Température"
                )
                num_sources = gr.Slider(
                    1, 10, value=5, step=1,
                    label="Nombre de sources médicales"
                )
            
            analyze_btn = gr.Button("🔬 Analyser", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Résultats")
            
            with gr.Tabs():
                with gr.Tab("� Sources Médicales"):
                    sources_output = gr.Textbox(
                        label="Littérature récupérée",
                        lines=12,
                        max_lines=25,
                        elem_id="sources_output"
                    )
                
                with gr.Tab("🏥 Diagnostic VLM"):
                    diagnosis_output = gr.Textbox(
                        label="Diagnostic clinique avec citations",
                        lines=12,
                        max_lines=25,
                        elem_id="diagnosis_output"
                    )
    
    with gr.Row():
        gr.Markdown("""
        ---
        ### 🎯 Instructions:
        
        1. **Téléchargez** une image de lésion cutanée
        2. **Ajoutez** du contexte clinique optionnel (âge, symptômes, durée, etc.)
        3. **Ajustez** les paramètres si nécessaire
        4. **Cliquez** sur "Analyser"
        5. **Consultez** les résultats dans les 2 onglets
        6. Le rapport complet est sauvegardé automatiquement (analysis_YYYYMMDD_HHMMSS.txt)
        
        ### ⚡ Note: Premier lancement
        Le premier lancement prend ~1-2 minutes (chargement des modèles).
        """)
    
    # Connecter le bouton
    analyze_btn.click(
        fn=analyze_lesion_complete,
        inputs=[
            image_input,
            additional_context,
            max_tokens,
            temperature,
            num_sources
        ],
        outputs=[sources_output, diagnosis_output]
    )


# ============================================================================
# Lancement
# ============================================================================

if __name__ == "__main__":
    gradio_config = config.get('gradio', {})
    
    logger.info("="*80)
    logger.info("🚀 Lancement de VLM-Bot Gradio App")
    logger.info("="*80)
    logger.info(f"   Port: {gradio_config.get('port', 7860)}")
    logger.info(f"   Share: {gradio_config.get('share', False)}")
    logger.info("="*80)
    
    demo.launch(
        server_name=gradio_config.get('server_name', '0.0.0.0'),
        server_port=gradio_config.get('port', 7860),
        share=gradio_config.get('share', False),
        debug=True,
        show_error=True
    )
