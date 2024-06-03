import gradio as gr
import numpy as np
import json
import tensorflow.lite as tflite

gr.set_static_paths(paths=["/app/atlas/"])  

gr.set_static_paths(paths=["/app/atlas/"])

js = json.load(open("atlas.json",'r'))
"""[{"path":"atlas/girl-8m.jpg","name":"girl-8m","gender":"girl","ageo":"8m","age":0.6666666667},"""
delta = 3
img_height, img_width = 256,256

interpreter = tflite.Interpreter(model_path="bone-age-densenet.tflite")

def inference(gender, image):
    print(type(image))
    image = image.resize( (img_width,img_height))
    sex = 1 if gender == 'boy' else 0    
    img = np.array(image)
    np_img = img/256    
    interpreter.allocate_tensors()
# Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape1 = input_details[1]['shape']
    input_shape0 = input_details[0]['shape']
    input_data1 = np.array(np_img , dtype=np.float32).reshape(input_shape1)
    input_data0 = np.array([float(sex)], dtype=np.float32).reshape(input_shape0)
    interpreter.set_tensor(input_details[1]['index'], input_data1)
    interpreter.set_tensor(input_details[0]['index'], input_data0)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    age_range = output_data [0][0]   
    print(age_range)
    
    output = f"predicted Bone Age about {age_range:.1f} y/o\ngender:{gender}"
    #print(s)
    pathlist = [j['path'] for j in js if 
                j["age"]>(age_range-delta) and j["age"]<(age_range+delta) and j['gender']==gender]
    return output, gr.Gallery(pathlist, label="Atlas", show_download_button=False, show_share_button=False)

demo = gr.Interface(
    fn=inference,
    inputs=[ gr.Radio(["boy","girl"], label='Gender', value='boy'), gr.Image(label="Image", type="pil", image_mode="L")],
    outputs=[gr.Text(label="result"),gr.Gallery(label="Atlas")],
    title="Bone Age",
    description=r"Bone age AI training with 8K+ bone age data base on [Tanner-Whitehouse mathod](http://vl.academicdirect.ro/medical_informatics/bone_age/v1.0/), inference using tflite",
    article=r"Due to the fact that the population mainly consists of 6 to 10 year-olds, accuracy outside this range may be limited. <br/>Disclaimer: This information is intended for general purposes only and should not be used for medical purposes. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment",
    allow_flagging = 'manual',
    flagging_options=['agree','disagree'] 
)
#model = keras.layers.TFSMLayer(r"pruned_bone_age.pb", call_endpoint='serving_default')

if __name__ == "__main__":
    demo.launch(allowed_paths=["/app/atlas","/image"], favicon_path="hello.png",server_port=80)
