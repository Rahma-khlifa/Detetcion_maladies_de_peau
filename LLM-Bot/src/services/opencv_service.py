"""
OpenCV Service - Extraction de caractéristiques visuelles
Analyse automatique des lésions dermatologiques.
"""

import cv2
import numpy as np
import imutils
from skimage import morphology
from sklearn.cluster import KMeans
import webcolors
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class OpenCVService:
    """Service pour l'extraction de caractéristiques OpenCV."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le service OpenCV.
        
        Args:
            config: Configuration OpenCV (dict depuis config.yaml)
        """
        self.config = config
        self.css3_colors = self._get_css3_names()
    
    def analyze_lesion(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Analyse complète d'une lésion dermatologique.
        
        Args:
            pil_image: Image PIL de la lésion
            
        Returns:
            Dictionnaire contenant toutes les mesures et la description textuelle
        """
        try:
            # Convertir et prétraiter
            img = self._load_and_preprocess(pil_image)
            img_no_hair = self._remove_hairs(img)
            
            # Segmenter la lésion
            mask = self._segment_lesion(img_no_hair)
            
            # Extraire les caractéristiques
            shape_features = self._compute_shape_features(mask)
            colors = self._analyze_colors(img_no_hair, mask)
            texture = self._analyze_texture(img_no_hair, mask)
            border = self._assess_border_quality(mask)
            
            # Distribution spatiale des couleurs
            color_distribution = self._analyze_color_distribution(
                img_no_hair, mask, colors
            )
            
            # Générer la description textuelle
            description = self._make_description(
                colors, shape_features, texture, border, color_distribution
            )
            
            return {
                'description': description,
                'shape': shape_features,
                'colors': colors,
                'texture': texture,
                'border': border,
                'color_distribution': color_distribution,
                'images': {
                    'original': img,
                    'no_hair': img_no_hair,
                    'mask': mask
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse OpenCV: {e}")
            return {'description': f"Erreur: {str(e)}", 'error': True}
    
    # ========================================================================
    # Preprocessing
    # ========================================================================
    
    def _load_and_preprocess(self, pil_image: Image.Image) -> np.ndarray:
        """Convertit PIL image en format OpenCV et redimensionne."""
        img = np.array(pil_image)
        
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        max_dim = self.config.get('max_image_dim', 1024)
        img = imutils.resize(img, width=min(max_dim, img.shape[1]))
        
        return img
    
    def _remove_hairs(self, img_rgb: np.ndarray) -> np.ndarray:
        """Supprime les artefacts de poils."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, th = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=1)
        
        inpaint = cv2.inpaint(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            th, 3, cv2.INPAINT_TELEA
        )
        
        return cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
    
    def _segment_lesion(self, img_rgb: np.ndarray) -> np.ndarray:
        """Segmente la lésion via k-means clustering."""
        k = self.config.get('segmentation_clusters', 3)
        min_size = self.config.get('min_lesion_size', 500)
        
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        X = lab.reshape((-1, 3)).astype(np.float32)
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        labels = km.labels_.reshape(img_rgb.shape[:2])
        
        # Trouver le cluster le plus sombre (lésion)
        cluster_means = []
        for i in range(k):
            mask_i = (labels == i)
            if mask_i.sum() > 0:
                cluster_means.append(lab[:, :, 0][mask_i].mean())
            else:
                cluster_means.append(999)
        
        lesion_label = int(np.argmin(cluster_means))
        mask = (labels == lesion_label).astype(bool)
        mask = morphology.remove_small_objects(mask, min_size=min_size)
        mask = morphology.remove_small_holes(mask, area_threshold=min_size)
        
        return (mask.astype('uint8') * 255)
    
    # ========================================================================
    # Feature Extraction
    # ========================================================================
    
    def _compute_shape_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Calcule les caractéristiques de forme (aire, périmètre, circularité, asymétrie)."""
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            return {}
        
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-8)
        (x, y, w, h) = cv2.boundingRect(c)
        diameter_px = max(w, h)
        
        # Calcul d'asymétrie
        asymmetry = self._calculate_asymmetry(c, mask)
        
        return {
            'area_px': float(area),
            'perimeter_px': float(perimeter),
            'circularity': float(circularity),
            'diameter_px': int(diameter_px),
            'asymmetry': asymmetry
        }
    
    def _calculate_asymmetry(self, contour: np.ndarray, mask: np.ndarray) -> float:
        """Calcule le score d'asymétrie."""
        ys = contour[:, :, 1].flatten()
        xs = contour[:, :, 0].flatten()
        cx, cy = xs.mean(), ys.mean()
        
        try:
            cov = np.cov(xs - cx, ys - cy)
            eigvals, eigvecs = np.linalg.eig(cov)
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        except:
            return 0.0
        
        # Rotation et comparaison des moitiés
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        h_mask, w_mask = mask.shape
        rotated = cv2.warpAffine(mask, M, (w_mask, h_mask),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        
        ys_nonzero, xs_nonzero = np.where(rotated > 0)
        if len(xs_nonzero) == 0:
            return 0.0
        
        minx, maxx = xs_nonzero.min(), xs_nonzero.max()
        miny, maxy = ys_nonzero.min(), ys_nonzero.max()
        crop = rotated[miny:maxy+1, minx:maxx+1]
        
        mid = crop.shape[1] // 2
        left = crop[:, :mid]
        right = crop[:, -mid:] if mid > 0 else np.zeros_like(left)
        
        if left.shape != right.shape:
            minw = min(left.shape[1], right.shape[1])
            left = left[:, :minw]
            right = right[:, -minw:]
        
        diff = np.sum(np.abs(left.astype(int) - np.fliplr(right).astype(int)))
        asym_score = diff / max(1, crop.size)
        
        return float(asym_score)
    
    def _analyze_colors(self, img_rgb: np.ndarray, mask: np.ndarray) -> List[Dict[str, Any]]:
        """Extrait les couleurs dominantes via k-means."""
        n_colors = self.config.get('n_colors', 4)
        pts = img_rgb[mask > 0].reshape(-1, 3)
        
        if pts.shape[0] == 0:
            return []
        
        n_clusters = min(n_colors, 6, pts.shape[0])
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pts)
        
        centers = km.cluster_centers_.astype(int)
        counts = np.bincount(km.labels_)
        order = np.argsort(-counts)
        
        results = []
        total = km.labels_.size
        for idx in order:
            rgb = tuple(int(x) for x in centers[idx].tolist())
            name = self._rgb_to_name(rgb)
            pct = 100.0 * counts[idx] / total
            results.append({
                'rgb': rgb,
                'name': name,
                'derm_category': self._classify_derm_color(rgb),
                'pct': float(pct)
            })
        
        return results
    
    def _analyze_color_distribution(
        self,
        img_rgb: np.ndarray,
        mask: np.ndarray,
        colors: List[Dict]
    ) -> Dict[int, Dict[str, float]]:
        """Analyse la distribution spatiale des couleurs."""
        if not colors:
            return {}
        
        pts = img_rgb[mask > 0].reshape(-1, 3)
        n_clusters = min(len(colors), 4, pts.shape[0])
        
        if n_clusters == 0:
            return {}
        
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(pts)
        
        ys, xs = np.where(mask > 0)
        center_y, center_x = ys.mean(), xs.mean()
        y_coords, x_coords = np.where(mask > 0)
        distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        max_dist = distances.max()
        
        distribution = {}
        for i in range(n_clusters):
            color_mask = (km.labels_ == i)
            color_distances = distances[color_mask]
            central_count = np.sum(color_distances < max_dist * 0.5)
            peripheral_count = np.sum(color_distances >= max_dist * 0.5)
            total = central_count + peripheral_count
            
            if total > 0:
                distribution[i] = {
                    'central_pct': 100 * central_count / total,
                    'peripheral_pct': 100 * peripheral_count / total
                }
        
        return distribution
    
    def _analyze_texture(self, img_rgb: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Analyse la texture et les motifs de pigmentation."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lesion_patch = gray[mask > 0]
        texture_variance = float(np.var(lesion_patch))
        
        edges = cv2.Canny(gray, 50, 150)
        lesion_edges = edges[mask > 0]
        edge_density = float(np.sum(lesion_edges > 0) / len(lesion_edges))
        
        # Classification du motif
        if edge_density > 0.15:
            pattern = "reticular/network pattern visible"
        elif edge_density > 0.05:
            pattern = "irregular pigmentation pattern"
        else:
            pattern = "homogeneous pigmentation"
        
        # Classification de surface
        if texture_variance > 1000:
            surface = "highly textured/varied"
        elif texture_variance > 500:
            surface = "moderately textured"
        else:
            surface = "smooth/uniform"
        
        return {
            'pattern': pattern,
            'surface': surface,
            'texture_variance': texture_variance,
            'edge_density': edge_density
        }
    
    def _assess_border_quality(self, mask: np.ndarray) -> Dict[str, Any]:
        """Évalue l'irrégularité et la définition des bordures."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return {}
        
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        irregularity_score = len(approx) / perimeter * 1000
        
        # Définition de bordure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        border_zone = dilated - eroded
        border_width = np.sum(border_zone > 0) / perimeter
        
        border_definition = "well-defined" if border_width < 3 else "poorly-defined"
        border_regularity = "regular" if irregularity_score < 5 else "irregular"
        if irregularity_score > 10:
            border_regularity = "highly irregular/notched"
        
        return {
            'definition': border_definition,
            'regularity': border_regularity,
            'irregularity_score': float(irregularity_score),
            'border_width': float(border_width)
        }
    
    # ========================================================================
    # Utilities
    # ========================================================================
    
    def _get_css3_names(self) -> Dict[str, str]:
        """Obtient le mapping des noms de couleurs CSS3."""
        try:
            return webcolors.CSS3_NAMES_TO_HEX
        except AttributeError:
            color_dict = {}
            for name in webcolors.names('css3'):
                try:
                    color_dict[name] = webcolors.name_to_hex(name, spec='css3')
                except ValueError:
                    pass
            return color_dict
    
    def _rgb_to_name(self, rgb_triplet: Tuple[int, int, int]) -> str:
        """Convertit RGB en nom de couleur CSS3 le plus proche."""
        try:
            return webcolors.rgb_to_name(tuple(int(x) for x in rgb_triplet), spec='css3')
        except ValueError:
            min_dist = None
            min_name = None
            r, g, b = rgb_triplet
            
            for name, hexv in self.css3_colors.items():
                rn, gn, bn = webcolors.hex_to_rgb(hexv)
                d = (r - rn) ** 2 + (g - gn) ** 2 + (b - bn) ** 2
                if min_dist is None or d < min_dist:
                    min_dist = d
                    min_name = name
            
            return min_name or "unknown"
    
    def _classify_derm_color(self, rgb: Tuple[int, int, int]) -> str:
        """Classifie la couleur selon la terminologie dermatologique."""
        r, g, b = rgb
        
        if r > 200 and g > 180 and b > 180:
            return "white/depigmented"
        if r > 150 and g < 100 and b < 100:
            return "red/erythematous"
        if r > 180 and g > 120 and b > 120:
            return "pink"
        if r > 100 and g > 60:
            if r > 150 and g > 100:
                return "light brown/tan"
            elif r > 100 and g > 60:
                return "medium brown"
            else:
                return "dark brown"
        if r < 60 and g < 60 and b < 60:
            return "black/very dark brown"
        if b > r and b > g and r < 100:
            return "blue-gray (regression)"
        
        return "brown"
    
    def _make_description(
        self,
        colors: List[Dict],
        shape: Dict,
        texture: Dict,
        border: Dict,
        color_distribution: Dict
    ) -> str:
        """Génère la description textuelle complète."""
        pixels_per_mm = self.config.get('pixels_per_mm', 10)
        diameter_mm = shape['diameter_px'] / pixels_per_mm
        area_mm2 = shape['area_px'] / (pixels_per_mm ** 2)
        
        # Description des couleurs avec distribution
        color_descs = []
        for i, c in enumerate(colors):
            dist = color_distribution.get(i, {})
            location = ""
            if dist:
                if dist['central_pct'] > 70:
                    location = " (predominantly central)"
                elif dist['peripheral_pct'] > 70:
                    location = " (predominantly peripheral)"
                else:
                    location = " (mixed distribution)"
            color_descs.append(f"{c['derm_category']} {c['pct']:.1f}%{location}")
        
        # Description d'asymétrie
        if shape['asymmetry'] > 40:
            asymmetry_desc = "markedly asymmetric"
        elif shape['asymmetry'] > 25:
            asymmetry_desc = "moderately asymmetric"
        else:
            asymmetry_desc = "relatively symmetric"
        
        # Facteurs de risque ABCDE
        risk_factors = self._calculate_abcde_risk(colors, shape, border, diameter_mm)
        
        description = f"""DERMATOLOGICAL LESION ANALYSIS:

MORPHOLOGY:
- Size: Approximately {diameter_mm:.1f}mm diameter, {area_mm2:.1f}mm² area
- Shape: {asymmetry_desc} with {border['regularity']} borders
- Border definition: {border['definition']}
- Overall circularity: {shape['circularity']:.3f} (1.0 = perfect circle)

COLOR ANALYSIS:
- Number of distinct color zones: {len(colors)}
- Color composition: {'; '.join(color_descs)}
- Color pattern: {'Variegated (multiple distinct colors)' if len(colors) >= 3 else 'Relatively uniform'}

SURFACE & TEXTURE:
- Pigmentation pattern: {texture['pattern']}
- Surface appearance: {texture['surface']}
- Texture complexity score: {texture['texture_variance']:.1f}

BORDER CHARACTERISTICS:
- Border regularity: {border['regularity']}
- Border definition: {border['definition']}
- Irregularity score: {border['irregularity_score']:.2f}

ASYMMETRY ASSESSMENT:
- Asymmetry score: {shape['asymmetry']:.1f}
- Classification: {asymmetry_desc}

ABCDE MELANOMA RISK FACTORS:
"""
        
        if risk_factors:
            for factor in risk_factors:
                description += f"⚠️  {factor}\n"
        else:
            description += "- No major ABCDE risk factors detected\n"
        
        description += f"""
SUMMARY FOR CLINICAL CORRELATION:
This lesion presents with {len(colors)} distinct color zones, {border['regularity']} borders,
and {asymmetry_desc} morphology. {'Multiple concerning features warrant further evaluation.' if len(risk_factors) >= 2 else 'Clinical correlation recommended for definitive diagnosis.'}

NOTE: This is an automated image analysis. Clinical evaluation by a dermatologist
is essential for accurate diagnosis and management decisions.
"""
        
        return description
    
    def _calculate_abcde_risk(
        self,
        colors: List[Dict],
        shape: Dict,
        border: Dict,
        size_mm: float
    ) -> List[str]:
        """Calcule les facteurs de risque ABCDE."""
        risk_factors = []
        
        # A - Asymétrie
        if shape['asymmetry'] > 30:
            risk_factors.append("A: Significant asymmetry detected")
        
        # B - Bordure irrégulière
        if border['irregularity_score'] > 8:
            risk_factors.append("B: Highly irregular border")
        elif shape['circularity'] < 0.3:
            risk_factors.append("B: Irregular border shape")
        
        # C - Variété de couleurs
        if len(colors) >= 4:
            risk_factors.append("C: Multiple colors present (≥4 distinct tones)")
        
        color_names = [c['derm_category'] for c in colors]
        if any('black' in c for c in color_names):
            risk_factors.append("C: Black pigmentation present")
        if any('blue-gray' in c for c in color_names):
            risk_factors.append("C: Blue-gray areas (possible regression)")
        
        # D - Diamètre
        if size_mm and size_mm > 6:
            risk_factors.append(f"D: Diameter > 6mm ({size_mm:.1f}mm)")
        
        return risk_factors
