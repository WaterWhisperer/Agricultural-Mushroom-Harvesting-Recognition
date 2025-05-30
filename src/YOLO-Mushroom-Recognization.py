import os
import json
from process import process_img  

def main():
    imgs_folder = './imgs/'
    output_file = 'output.json'
    
    img_paths = [os.path.join(imgs_folder, img) for img in os.listdir(imgs_folder) if img.endswith('.jpg')]
    
    results = {}
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        mushroom_list = process_img(img_path)
        results[img_name] = mushroom_list
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detection complete. Results saved to {output_file}")

if __name__ == '__main__':
    main()