"""
Application Gradio - LLM-Bot
Interface web pour l'analyse dermatologique basée sur texte (Flan-T5-XL + RAG) + OpenCV.
"""

import gradio as gr
import logging
from pathlib import Path
from datetime import datetime
from PIL import Image

from services.llm_service import LLMService
from services.rag_service import RAGService
from services.opencv_service import OpenCVService
from utils.helpers import (
    load_config,
    setup_logging,
    load_environment,
    ensure_directories,
    format_prompt,
    format_medical_prompt
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
llm_service = None
rag_service = None
opencv_service = None


def initialize_services():
    """Initialise les services LLM, RAG et OpenCV (lazy loading)."""
    global llm_service, rag_service, opencv_service
    
    if llm_service is None:
        logger.info("🔄 Initialisation du LLM...")
        llm_service = LLMService(config['models']['llm'])
        llm_service.load_model()
    
    if rag_service is None:
        logger.info("🔄 Chargement de l'index RAG...")
        rag_service = RAGService(config['rag'])
        index_path = config['rag'].get('index_path', 'data/processed/faiss_index')
        
        if Path(index_path).exists():
            rag_service.load_index(index_path)
        else:
            logger.warning("⚠️  Index RAG non trouvé. Construction en cours...")
            rag_service.build_index(save_path=index_path)

    if opencv_service is None:
        logger.info("🔄 Initialisation d'OpenCV...")
        opencv_service = OpenCVService(config.get('opencv', {}))


def analyze_symptoms(
    symptoms: str,
    duration: str,
    use_rag: bool,
    num_sources: int,
    max_tokens: int,
    temperature: float
):
    """
    Analyse basée sur texte (symptômes).
    
    Args:
        symptoms: Description des symptômes
        duration: Durée des symptômes
        use_rag: Utiliser RAG pour contexte médical
        num_sources: Nombre de sources RAG
        max_tokens: Max tokens à générer
        temperature: Température de sampling
        
    Returns:
        Tuple (sources_text, diagnosis_text)
    """
    if not symptoms or not symptoms.strip():
        return "⚠️ Veuillez décrire les symptômes!", ""
    
    try:
        # Initialiser les services (lazy)
        initialize_services()
        
        # Phase 1: Recherche RAG
        sources_text = ""
        retrieved_context = ""
        
        if use_rag:
            logger.info("📚 Recherche RAG...")
            
            # Construire query à partir des symptômes
            query_text = f"{symptoms} {duration}"
            rag_results = rag_service.search(query_text, top_k=int(num_sources))
            
            sources_text = f"**Found {len(rag_results)} relevant medical abstracts:**\n\n"
            
            for i, (doc, score) in enumerate(rag_results, 1):
                sources_text += f"**[Source {i}]** (Relevance: {score:.4f})\n"
                sources_text += f"{doc.page_content}\n"
                sources_text += f"{'-'*80}\n\n"
                retrieved_context += f"\n[Source {i}]:\n{doc.page_content}\n"
        else:
            sources_text = "**RAG désactivé** - Génération basée uniquement sur les connaissances du LLM.\n"
        
        # Phase 2: Construire le prompt
        if use_rag and retrieved_context:
            prompt = format_medical_prompt(symptoms, duration, retrieved_context)
        else:
            prompt = format_prompt(
                f"Patient symptoms: {symptoms}\nDuration: {duration}",
                "",
                mode="direct"
            )
        
        # Phase 3: Génération LLM
        logger.info("🤖 Génération de l'analyse...")
        diagnosis = llm_service.generate_response(
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        
        # Sauvegarder le rapport
        if use_rag and retrieved_context:
            rag_block = f"RETRIEVED MEDICAL LITERATURE:\n{retrieved_context}\n\n{'='*80}\n"
        else:
            rag_block = ""
        
        report = f"""
DERMATOLOGICAL ANALYSIS REPORT (TEXT-BASED)
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: Flan-T5-XL + RAG

PATIENT INFORMATION:
- Symptoms: {symptoms}
- Duration: {duration}

{rag_block}

PRELIMINARY ASSESSMENT:
{diagnosis}

DISCLAIMER: For research and educational purposes only. NOT a substitute for
professional medical advice, diagnosis, or treatment. Consult a qualified dermatologist.
{'='*80}
"""
        
        filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Analyse terminée. Rapport: {filename}")
        
        return sources_text, diagnosis
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""


def answer_question(
    question: str,
    use_rag: bool,
    num_sources: int,
    max_tokens: int,
    temperature: float
):
    """
    Répond à une question médicale générale.
    
    Args:
        question: Question de l'utilisateur
        use_rag: Utiliser RAG
        num_sources: Nombre de sources
        max_tokens: Tokens max
        temperature: Température
        
    Returns:
        Tuple (sources_text, answer_text)
    """
    if not question or not question.strip():
        return "⚠️ Veuillez poser une question!", ""
    
    try:
        initialize_services()
        
        # RAG search
        sources_text = ""
        retrieved_context = ""
        
        if use_rag:
            logger.info("📚 Recherche RAG...")
            rag_results = rag_service.search(question, top_k=int(num_sources))
            
            sources_text = f"**Found {len(rag_results)} relevant sources:**\n\n"
            
            for i, (doc, score) in enumerate(rag_results, 1):
                sources_text += f"**[Source {i}]** (Relevance: {score:.4f})\n"
                sources_text += f"{doc.page_content}\n"
                sources_text += f"{'-'*80}\n\n"
                retrieved_context += f"\n[Source {i}]:\n{doc.page_content}\n"
        else:
            sources_text = "**RAG désactivé**\n"
        
        # Prompt
        if use_rag and retrieved_context:
            prompt = format_prompt(question, retrieved_context, mode="with_context")
        else:
            prompt = format_prompt(question, "", mode="direct")
        
        # Generate
        logger.info("🤖 Génération de la réponse...")
        answer = llm_service.generate_response(
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        
        return sources_text, answer
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, ""


def analyze_image(
    image: Image.Image,
    use_rag: bool,
    num_sources: int,
    max_tokens: int,
    temperature: float
):
    """
    Analyse d'image dermatologique avec OpenCV + LLM.
    
    Args:
        image: Image PIL uploadée
        use_rag: Utiliser RAG
        num_sources: Nombre de sources
        max_tokens: Tokens max
        temperature: Température
        
    Returns:
        Tuple (opencv_analysis, sources_text, llm_diagnosis)
    """
    if image is None:
        return "⚠️ Veuillez uploader une image!", "", ""
    
    try:
        # Initialiser les services
        initialize_services()
        
        # Phase 1: Analyse OpenCV
        logger.info("🔬 Analyse OpenCV de l'image...")
        opencv_result = opencv_service.analyze_lesion(image)
        
        if opencv_result.get('error'):
            return opencv_result['description'], "", ""
        
        opencv_description = opencv_result['description']
        
        # Phase 2: Recherche RAG basée sur l'analyse OpenCV
        sources_text = ""
        retrieved_context = ""
        
        if use_rag:
            logger.info("📚 Recherche RAG basée sur l'analyse visuelle...")
            
            # Construire query à partir des caractéristiques OpenCV
            shape = opencv_result.get('shape', {})
            colors = opencv_result.get('colors', [])
            border = opencv_result.get('border', {})
            
            query_parts = []
            
            # Asymétrie
            if shape.get('asymmetry', 0) > 30:
                query_parts.append("asymmetric lesion")
            
            # Bordure
            if border.get('regularity') in ['irregular', 'highly irregular/notched']:
                query_parts.append("irregular border")
            
            # Couleurs
            if len(colors) >= 3:
                query_parts.append("multiple colors variegated")
            color_categories = [c.get('derm_category', '') for c in colors]
            if any('black' in cat for cat in color_categories):
                query_parts.append("black pigmentation")
            
            # Diamètre
            diameter_mm = shape.get('diameter_px', 0) / config.get('opencv', {}).get('pixels_per_mm', 10)
            if diameter_mm > 6:
                query_parts.append("large lesion melanoma")
            
            query_text = " ".join(query_parts) if query_parts else "skin lesion dermatology"
            
            rag_results = rag_service.search(query_text, top_k=int(num_sources))
            
            sources_text = f"**Found {len(rag_results)} relevant medical sources:**\n\n"
            
            for i, (doc, score) in enumerate(rag_results, 1):
                sources_text += f"**[Source {i}]** (Relevance: {score:.4f})\n"
                sources_text += f"{doc.page_content}\n"
                sources_text += f"{'-'*80}\n\n"
                retrieved_context += f"\n[Source {i}]:\n{doc.page_content}\n"
        else:
            sources_text = "**RAG désactivé**\n"
        
        # Phase 3: Génération LLM basée sur OpenCV + RAG
        logger.info("🤖 Génération du diagnostic LLM...")
        
        # Construire le prompt combinant OpenCV et RAG
        if use_rag and retrieved_context:
            prompt = f"""Based on the following automated image analysis and medical literature, provide a clinical interpretation and differential diagnosis:

AUTOMATED IMAGE ANALYSIS:
{opencv_description}

RELEVANT MEDICAL LITERATURE:
{retrieved_context}

Provide a clinical interpretation including:
1. Summary of key findings
2. Differential diagnosis (most likely conditions)
3. Recommended next steps
4. Red flags or concerning features

Clinical Interpretation:"""
        else:
            prompt = f"""Based on the following automated image analysis, provide a clinical interpretation:

AUTOMATED IMAGE ANALYSIS:
{opencv_description}

Provide a clinical interpretation including differential diagnosis and recommendations.

Clinical Interpretation:"""
        
        diagnosis = llm_service.generate_response(
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        
        # Sauvegarder le rapport complet
        if use_rag and retrieved_context:
            rag_block = f"RETRIEVED MEDICAL LITERATURE:\n{retrieved_context}\n\n{'='*80}\n"
        else:
            rag_block = ""
        
        report = f"""
COMPREHENSIVE DERMATOLOGICAL IMAGE ANALYSIS
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analysis Method: OpenCV Feature Extraction + Flan-T5-XL + RAG

{'='*80}
AUTOMATED IMAGE ANALYSIS (OpenCV):
{'='*80}
{opencv_description}

{'='*80}
{rag_block}

{'='*80}
CLINICAL INTERPRETATION (LLM):
{'='*80}
{diagnosis}

{'='*80}
DISCLAIMER: For research and educational purposes only. NOT a substitute for
professional medical advice, diagnosis, or treatment. Consult a qualified dermatologist.
{'='*80}
"""
        
        filename = f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"✅ Analyse d'image terminée. Rapport: {filename}")
        
        return opencv_description, sources_text, diagnosis
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse image: {e}", exc_info=True)
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", ""


def analyze_case(
    symptoms: str,
    duration: str,
    image: Image.Image,
    use_rag: bool,
    num_sources: int,
    max_tokens: int,
    temperature: float
):
    """
    Analyse combinée texte + image : utilise OpenCV pour extraire les caractéristiques visuelles,
    optionnellement RAG pour le contexte, puis génère une interprétation clinique via LLM.

    Returns (opencv_description, sources_text, diagnosis)
    """
    if (not symptoms or not symptoms.strip()) and image is None:
        return "⚠️ Veuillez fournir une description ou une image!", "", ""

    try:
        initialize_services()

        # OpenCV analysis (if image provided)
        opencv_description = ""
        if image is not None:
            logger.info("🔬 Analyse OpenCV (combinée)...")
            opencv_result = opencv_service.analyze_lesion(image)
            if opencv_result.get('error'):
                return opencv_result['description'], "", ""
            opencv_description = opencv_result.get('description', '')

        # Build RAG query from symptoms + visual summary
        sources_text = ""
        retrieved_context = ""

        if use_rag:
            logger.info("📚 Recherche RAG (combinée)...")
            query_parts = []
            if symptoms and symptoms.strip():
                query_parts.append(symptoms)
                if duration and duration.strip():
                    query_parts.append(duration)
            if opencv_description:
                query_parts.append(opencv_description)

            query_text = " ".join(query_parts) if query_parts else "skin lesion dermatology"

            rag_results = rag_service.search(query_text, top_k=int(num_sources))
            sources_text = f"**Found {len(rag_results)} relevant medical sources:**\n\n"
            for i, (doc, score) in enumerate(rag_results, 1):
                sources_text += f"**[Source {i}]** (Relevance: {score:.4f})\n"
                sources_text += f"{doc.page_content}\n"
                sources_text += f"{'-'*80}\n\n"
                retrieved_context += f"\n[Source {i}]:\n{doc.page_content}\n"
        else:
            sources_text = "**RAG désactivé**\n"

        # Build combined prompt
        prompt_parts = ["Based on the following patient information, automated image analysis (if any), and medical literature, provide a clinical interpretation and differential diagnosis:\n"]
        prompt_parts.append(f"PATIENT SYMPTOMS:\n{symptoms if symptoms else 'N/A'}\nDuration: {duration if duration else 'N/A'}\n")
        prompt_parts.append("AUTOMATED IMAGE ANALYSIS:\n")
        prompt_parts.append(opencv_description or "No image analysis available.")

        if retrieved_context:
            prompt_parts.append("\nRELEVANT MEDICAL LITERATURE:\n")
            prompt_parts.append(retrieved_context)

        prompt_parts.append("\nProvide:\n1) Summary of key findings\n2) Differential diagnosis (most likely conditions)\n3) Recommended next steps and investigations\n4) Red flags or urgent features\n\nClinical Interpretation:")

        prompt = "\n".join(prompt_parts)

        logger.info("🤖 Génération du diagnostic combiné...")
        diagnosis = llm_service.generate_response(
            prompt=prompt,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature)
        )

        # Save a combined report
        report = f"""
COMBINED DERMATOLOGICAL ANALYSIS
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT SYMPTOMS:
{symptoms if symptoms else 'N/A'}
Duration: {duration if duration else 'N/A'}

AUTOMATED IMAGE ANALYSIS (OpenCV):
{opencv_description}

RETRIEVED MEDICAL LITERATURE:
{retrieved_context}

CLINICAL INTERPRETATION (LLM):
{diagnosis}

DISCLAIMER: For research and educational purposes only. NOT a substitute for professional medical advice.
{'='*80}
"""

        filename = f"combined_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✅ Analyse combinée terminée. Rapport: {filename}")

        return opencv_description, sources_text, diagnosis

    except Exception as e:
        logger.error(f"❌ Erreur analyse combinée: {e}", exc_info=True)
        import traceback
        error_msg = f"❌ Erreur: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", ""


# ============================================================================
# Interface Gradio
# ============================================================================

custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .output-text {
        max-height: 500px;
        overflow-y: auto;
    }
"""

with gr.Blocks(
    title="LLM-Bot - Dermatological Q&A",
) as demo:
    
    gr.Markdown("""
    # 🤖 LLM-Bot - Système de Questions/Réponses Dermatologiques
    
    **Flan-T5-XL + RAG + OpenCV**
    
    - 🤖 **LLM**: Google Flan-T5-XL avec quantisation 8-bit
    - 📚 **RAG**: Recherche dans la littérature médicale
    - 🔬 **OpenCV**: Extraction automatique de caractéristiques visuelles
    - 💬 **Interface**: Analyse d'images, symptômes textuels et questions générales
    
    ⚠️ **DISCLAIMER**: Usage éducatif uniquement. Consultez toujours un dermatologue.
    """)
    
    with gr.Tabs():
        # ============================================================
        # Tab 1: Analyse d'Image (OpenCV + LLM)
        # ============================================================
        with gr.Tab("🔬 Analyse d'Image"):
            gr.Markdown("### Analysez une image de lésion dermatologique")
            gr.Markdown("OpenCV extrait automatiquement les caractéristiques visuelles (ABCDE), puis le LLM fournit une interprétation clinique.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="Image de la lésion",
                        type="pil",
                        height=400
                    )
                    
                    use_rag_image = gr.Checkbox(
                        value=True,
                        label="✅ Utiliser RAG (Recommandé)",
                        info="Recherche dans la littérature basée sur les caractéristiques visuelles"
                    )
                    
                    with gr.Accordion("⚙️ Paramètres avancés", open=False):
                        num_sources_image = gr.Slider(
                            1, 10, value=5, step=1,
                            label="Nombre de sources"
                        )
                        max_tokens_image = gr.Slider(
                            128, 1024, value=512, step=64,
                            label="Tokens de génération"
                        )
                        temperature_image = gr.Slider(
                            0.1, 1.0, value=0.7, step=0.1,
                            label="Température"
                        )
                    
                    analyze_image_btn = gr.Button("🔬 Analyser l'Image", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Résultats")
                    
                    with gr.Tabs():
                        with gr.Tab("🔬 Analyse OpenCV"):
                            opencv_output = gr.Textbox(
                                label="Caractéristiques visuelles extraites",
                                lines=15,
                                max_lines=30,
                                elem_classes=["output-text"]
                            )
                        
                        with gr.Tab("📚 Sources Médicales"):
                            sources_output_image = gr.Textbox(
                                label="Littérature récupérée",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
                        
                        with gr.Tab("🏥 Interprétation Clinique"):
                            diagnosis_output_image = gr.Textbox(
                                label="Diagnostic LLM",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
            
            analyze_image_btn.click(
                fn=analyze_image,
                inputs=[
                    image_input,
                    use_rag_image,
                    num_sources_image,
                    max_tokens_image,
                    temperature_image
                ],
                outputs=[opencv_output, sources_output_image, diagnosis_output_image]
            )
        # ============================================================
        # Tab: Analyse Combinée (Texte + Image)
        # ============================================================
        with gr.Tab("🧾 Analyse Combinée"):
            gr.Markdown("### Fournissez une description textuelle et/ou une image")

            with gr.Row():
                with gr.Column(scale=1):
                    combined_symptoms = gr.Textbox(
                        label="Symptômes (optionnel)",
                        placeholder="Ex: Lésion pigmentée avec bords irréguliers...",
                        lines=4
                    )

                    combined_duration = gr.Textbox(
                        label="Durée (optionnel)",
                        placeholder="Ex: 3 mois",
                        lines=1
                    )

                    combined_image = gr.Image(
                        label="Image de la lésion (optionnel)",
                        type="pil",
                        height=300
                    )

                    use_rag_combined = gr.Checkbox(
                        value=True,
                        label="✅ Utiliser RAG (Recommandé)"
                    )

                    with gr.Accordion("⚙️ Paramètres avancés", open=False):
                        num_sources_combined = gr.Slider(
                            1, 10, value=5, step=1,
                            label="Nombre de sources"
                        )
                        max_tokens_combined = gr.Slider(
                            128, 1024, value=512, step=64,
                            label="Tokens de génération"
                        )
                        temperature_combined = gr.Slider(
                            0.1, 1.0, value=0.7, step=0.1,
                            label="Température"
                        )

                    analyze_combined_btn = gr.Button("🧾 Analyser (Texte+Image)", variant="primary", size="lg")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Résultats")
                    with gr.Tabs():
                        with gr.Tab("🔬 Analyse OpenCV"):
                            opencv_output_combined = gr.Textbox(
                                label="Caractéristiques visuelles extraites",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )

                        with gr.Tab("📚 Sources Médicales"):
                            sources_output_combined = gr.Textbox(
                                label="Littérature récupérée",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )

                        with gr.Tab("🏥 Interprétation Clinique"):
                            diagnosis_output_combined = gr.Textbox(
                                label="Diagnostic LLM",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )

            analyze_combined_btn.click(
                fn=analyze_case,
                inputs=[
                    combined_symptoms,
                    combined_duration,
                    combined_image,
                    use_rag_combined,
                    num_sources_combined,
                    max_tokens_combined,
                    temperature_combined
                ],
                outputs=[opencv_output_combined, sources_output_combined, diagnosis_output_combined]
            )
        # ============================================================
        # Tab 2: Analyse de Symptômes
        # ============================================================
        with gr.Tab("🩺 Analyse de Symptômes"):
            gr.Markdown("### Décrivez les symptômes dermatologiques")
            
            with gr.Row():
                with gr.Column(scale=1):
                    symptoms_input = gr.Textbox(
                        label="Symptômes",
                        placeholder="Ex: Lésion pigmentée avec bords irréguliers...",
                        lines=5
                    )
                    
                    duration_input = gr.Textbox(
                        label="Durée",
                        placeholder="Ex: 3 mois",
                        lines=1
                    )
                    
                    use_rag_symptoms = gr.Checkbox(
                        value=True,
                        label="✅ Utiliser RAG (Recommandé)",
                        info="Recherche dans la littérature médicale"
                    )
                    
                    with gr.Accordion("⚙️ Paramètres avancés", open=False):
                        num_sources_symptoms = gr.Slider(
                            1, 10, value=5, step=1,
                            label="Nombre de sources"
                        )
                        max_tokens_symptoms = gr.Slider(
                            128, 1024, value=512, step=64,
                            label="Tokens de génération"
                        )
                        temperature_symptoms = gr.Slider(
                            0.1, 1.0, value=0.7, step=0.1,
                            label="Température"
                        )
                    
                    analyze_btn = gr.Button("🩺 Analyser", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Résultats")
                    
                    with gr.Tabs():
                        with gr.Tab("📚 Sources Médicales"):
                            sources_output_symptoms = gr.Textbox(
                                label="Littérature récupérée",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
                        
                        with gr.Tab("🏥 Analyse"):
                            diagnosis_output_symptoms = gr.Textbox(
                                label="Analyse préliminaire",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
            
            analyze_btn.click(
                fn=analyze_symptoms,
                inputs=[
                    symptoms_input,
                    duration_input,
                    use_rag_symptoms,
                    num_sources_symptoms,
                    max_tokens_symptoms,
                    temperature_symptoms
                ],
                outputs=[sources_output_symptoms, diagnosis_output_symptoms]
            )
        
        # ============================================================
        # Tab 2: Questions Générales
        # ============================================================
        with gr.Tab("❓ Questions Générales"):
            gr.Markdown("### Posez une question sur la dermatologie")
            
            with gr.Row():
                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Votre question",
                        placeholder="Ex: Quels sont les critères ABCDE pour le mélanome?",
                        lines=4
                    )
                    
                    use_rag_qa = gr.Checkbox(
                        value=True,
                        label="✅ Utiliser RAG (Recommandé)"
                    )
                    
                    with gr.Accordion("⚙️ Paramètres avancés", open=False):
                        num_sources_qa = gr.Slider(
                            1, 10, value=5, step=1,
                            label="Nombre de sources"
                        )
                        max_tokens_qa = gr.Slider(
                            128, 1024, value=512, step=64,
                            label="Tokens de génération"
                        )
                        temperature_qa = gr.Slider(
                            0.1, 1.0, value=0.7, step=0.1,
                            label="Température"
                        )
                    
                    ask_btn = gr.Button("❓ Demander", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Réponse")
                    
                    with gr.Tabs():
                        with gr.Tab("📚 Sources"):
                            sources_output_qa = gr.Textbox(
                                label="Sources médicales",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
                        
                        with gr.Tab("💡 Réponse"):
                            answer_output_qa = gr.Textbox(
                                label="Réponse détaillée",
                                lines=12,
                                max_lines=25,
                                elem_classes=["output-text"]
                            )
            
            ask_btn.click(
                fn=answer_question,
                inputs=[
                    question_input,
                    use_rag_qa,
                    num_sources_qa,
                    max_tokens_qa,
                    temperature_qa
                ],
                outputs=[sources_output_qa, answer_output_qa]
            )
    
    gr.Markdown("""
    ---
    ### 🎯 Instructions:
    
    **Analyse d'Image:**
    1. Uploadez une image de lésion dermatologique
    2. OpenCV analyse automatiquement les caractéristiques ABCDE
    3. RAG recherche dans la littérature médicale basée sur les caractéristiques détectées
    4. Le LLM génère une interprétation clinique complète
    5. Cliquez sur "Analyser l'Image"
    
    **Analyse de Symptômes:**
    1. Décrivez les symptômes observés (lésion, couleur, forme, etc.)
    2. Indiquez la durée
    3. Activez RAG pour une analyse basée sur la littérature
    4. Cliquez sur "Analyser"
    
    **Questions Générales:**
    1. Posez une question sur un sujet dermatologique
    2. Activez RAG pour des réponses citant la littérature
    3. Cliquez sur "Demander"
    
    ### ⚡ Note:
    Le premier lancement prend ~1-2 minutes (chargement des modèles).
    Les rapports sont sauvegardés automatiquement (analysis_YYYYMMDD_HHMMSS.txt / image_analysis_YYYYMMDD_HHMMSS.txt).
    """)


# ============================================================================
# Lancement
# ============================================================================

if __name__ == "__main__":
    gradio_config = config.get('gradio', {})
    
    logger.info("="*80)
    logger.info("🚀 Lancement de LLM-Bot Gradio App")
    logger.info("="*80)
    logger.info(f"   Port: {gradio_config.get('port', 7861)}")
    logger.info(f"   Share: {gradio_config.get('share', False)}")
    logger.info("="*80)
    
    demo.launch(
        server_name=gradio_config.get('server_name', '0.0.0.0'),
        server_port=gradio_config.get('port', 7861),
        share=gradio_config.get('share', False),
        debug=True,
        show_error=True
    )
