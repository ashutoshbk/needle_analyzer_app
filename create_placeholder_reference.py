"""
Create a placeholder reference image for hand rotation analysis
Run this script to generate a simple reference image if you don't have one yet

Usage:
    python create_placeholder_reference.py
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder_reference():
    # Create image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw border
    draw.rectangle([10, 10, width-10, height-10], outline='#667eea', width=5)
    
    # Title
    title_font_size = 40
    try:
        font_title = ImageFont.truetype("arial.ttf", title_font_size)
        font_text = ImageFont.truetype("arial.ttf", 24)
        font_label = ImageFont.truetype("arial.ttf", 20)
    except:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # Draw title
    title = "Hand Rotation Reference"
    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 30), title, fill='#333333', font=font_title)
    
    # Draw sections
    section_y = 120
    
    # Pronation section (left)
    pronation_x = 100
    draw.rectangle([pronation_x, section_y, pronation_x + 250, section_y + 200], 
                   outline='#dc3545', width=3)
    draw.text((pronation_x + 50, section_y + 20), "PRONATED", fill='#dc3545', font=font_text)
    draw.text((pronation_x + 20, section_y + 60), "Palm facing DOWN", fill='#666666', font=font_label)
    draw.text((pronation_x + 20, section_y + 90), "Rotation < 90°", fill='#666666', font=font_label)
    
    # Draw arrow pointing down (palm down indicator)
    arrow_x = pronation_x + 125
    arrow_y = section_y + 130
    draw.polygon([(arrow_x, arrow_y), (arrow_x-15, arrow_y-25), (arrow_x+15, arrow_y-25)], 
                 fill='#dc3545')
    
    # Supination section (right)
    supination_x = 450
    draw.rectangle([supination_x, section_y, supination_x + 250, section_y + 200], 
                   outline='#17a2b8', width=3)
    draw.text((supination_x + 40, section_y + 20), "SUPINATED", fill='#17a2b8', font=font_text)
    draw.text((supination_x + 20, section_y + 60), "Palm facing UP", fill='#666666', font=font_label)
    draw.text((supination_x + 20, section_y + 90), "Rotation > 90°", fill='#666666', font=font_label)
    
    # Draw arrow pointing up (palm up indicator)
    arrow_x = supination_x + 125
    arrow_y = section_y + 130
    draw.polygon([(arrow_x, arrow_y), (arrow_x-15, arrow_y+25), (arrow_x+15, arrow_y+25)], 
                 fill='#17a2b8')
    
    # Bottom notes
    notes_y = section_y + 240
    draw.text((50, notes_y), "• Rotation angle measured from initial hand position", 
              fill='#666666', font=font_label)
    draw.text((50, notes_y + 30), "• State transitions marked by vertical lines in graph", 
              fill='#666666', font=font_label)
    draw.text((50, notes_y + 60), "• Background shading indicates current state", 
              fill='#666666', font=font_label)
    
    # Footer
    footer = "Replace this with your custom reference image"
    footer_bbox = draw.textbbox((0, 0), footer, font=font_label)
    footer_width = footer_bbox[2] - footer_bbox[0]
    draw.text(((width - footer_width) // 2, height - 40), footer, 
              fill='#999999', font=font_label)
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), 'static', 'images', 'rotation_reference.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"✅ Created placeholder reference image: {output_path}")
    print(f"   You can replace this with your own reference image.")

if __name__ == "__main__":
    create_placeholder_reference()
