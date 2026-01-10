"""
SoM (Set-of-Mark) Annotator for WebCane3.
Draws numbered boxes on interactive elements for vision model analysis.

Uses SEQUENTIAL numbering (0, 1, 2...) for easier vision detection,
but returns a mapping of display_index -> element_id for accurate clicking.
"""

from PIL import Image, ImageDraw, ImageFont
import io
from typing import List, Dict, Tuple, Optional


class SoMAnnotator:
    """
    Annotate screenshots with numbered boxes for vision-based automation.
    
    Uses sequential numbering on the image (easier for Vision models to read)
    but maintains a mapping to original element IDs for accurate execution.
    """
    
    def __init__(self):
        """Initialize drawing parameters."""
        self.box_color = "#FF0000"  # Red boxes
        self.text_color = "#FFFFFF"  # White text
        self.bg_color = "#FF0000"    # Red background for labels
        self.font_size = 14
        self.font = self._load_font()
        
        # Mapping: display_index -> element_id
        # Updated each time annotate() is called
        self.index_to_id_map = {}
    
    def _load_font(self) -> ImageFont.FreeTypeFont:
        """Load font for text annotations with fallback."""
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        for path in font_paths:
            try:
                return ImageFont.truetype(path, self.font_size)
            except:
                continue
        
        # Fallback to default
        return ImageFont.load_default()
    
    def annotate(
        self, 
        screenshot_bytes: bytes, 
        elements: List[Dict],
        max_elements: int = 80
    ) -> Tuple[bytes, List[Dict]]:
        """
        Filter elements and annotate screenshot with numbered boxes.
        
        Uses sequential display indices (0, 1, 2...) on the image for easier
        Vision model reading, but maintains index_to_id_map for conversion.
        
        Args:
            screenshot_bytes: PNG screenshot as bytes
            elements: List of DOM elements
            max_elements: Maximum number of elements to annotate
            
        Returns:
            Tuple of (annotated image bytes, filtered elements list)
        """
        # Load image
        image = Image.open(io.BytesIO(screenshot_bytes))
        draw = ImageDraw.Draw(image)
        
        img_width, img_height = image.size
        
        # Reset mapping
        self.index_to_id_map = {}
        
        # Filter elements
        filtered = []
        for el in elements:
            bbox = el['bbox']
            
            # Skip tiny elements
            if bbox['w'] < 10 or bbox['h'] < 10:
                continue
            
            # Skip elements outside viewport
            if bbox['x'] < 0 or bbox['y'] < 0:
                continue
            if bbox['x'] + bbox['w'] > img_width or bbox['y'] + bbox['h'] > img_height:
                continue
            
            filtered.append(el)
            
            if len(filtered) >= max_elements:
                break
        
        # Draw boxes with SEQUENTIAL numbering (0, 1, 2...)
        # But store mapping: display_index -> element_id
        for display_idx, el in enumerate(filtered):
            self.index_to_id_map[display_idx] = el['id']
            self._draw_box(draw, el, str(display_idx))
        
        # Print mapping for debugging
        print(f"[SoM] Created {len(filtered)} annotations with index->id mapping")
        
        # Save to bytes
        output = io.BytesIO()
        image.save(output, format='PNG')
        annotated_bytes = output.getvalue()
        
        return annotated_bytes, filtered
    
    def get_element_id(self, display_index: int) -> int:
        """
        Convert a display index (from Vision agent) to actual element ID.
        
        Args:
            display_index: The number shown in the annotated image
            
        Returns:
            The actual element ID for clicking, or -1 if not found
        """
        element_id = self.index_to_id_map.get(display_index, -1)
        if element_id >= 0:
            print(f"[SoM] Mapping: display index {display_index} -> element ID {element_id}")
        else:
            print(f"[SoM] WARNING: No mapping for display index {display_index}")
        return element_id
    
    def _draw_box(self, draw: ImageDraw.ImageDraw, element: Dict, label: str):
        """
        Draw a labeled box around an element.
        
        Args:
            draw: PIL ImageDraw object
            element: Element dict with bbox
            label: Label text (the display index)
        """
        bbox = element['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        
        # Draw rectangle outline
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            outline=self.box_color,
            width=2
        )
        
        # Calculate label position (top-left, outside box if possible)
        label_bbox = draw.textbbox((0, 0), label, font=self.font)
        label_width = label_bbox[2] - label_bbox[0]
        label_height = label_bbox[3] - label_bbox[1]
        
        padding = 2
        label_x = x
        label_y = y - label_height - padding * 2
        
        # If label would be above viewport, put it inside
        if label_y < 0:
            label_y = y + padding
        
        # Draw label background
        draw.rectangle(
            [
                (label_x, label_y),
                (label_x + label_width + padding * 2, label_y + label_height + padding * 2)
            ],
            fill=self.bg_color
        )
        
        # Draw label text
        draw.text(
            (label_x + padding, label_y + padding),
            label,
            fill=self.text_color,
            font=self.font
        )
