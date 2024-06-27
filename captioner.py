import os, sys, csv
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def main(path, batch_size, output):
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    images = sorted([f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg', '.webm'))])
    prompt = "<MORE_DETAILED_CAPTION>"

    with open(output, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        for i in range(0, len(images), batch_size):
            image_files = images[i:i + batch_size]
            images_to_process = []

            for image_file in image_files:
                image_path = os.path.join(path, image_file)
                image = Image.open(image_path)
                images_to_process.append(image)

            inputs = processor(text=[prompt]*len(images_to_process), images=images_to_process, return_tensors="pt")

            generated_ids = model.generate(
                input_ids=inputs["input_ids"].cuda(),
                pixel_values=inputs["pixel_values"].cuda(),
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for image_file, generated_text in zip(image_files, generated_texts):
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task="<MORE_DETAILED_CAPTION>",
                    image_size=(images_to_process[0].width, images_to_process[0].height)  
                )

                output = str(parsed_answer)
                csv_writer.writerow([image_file, output[29:-2]])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_dir>")
    else:
        path = sys.argv[1]
        batch_size = 3
        output = 'output.csv'
        main(path, batch_size, output)
