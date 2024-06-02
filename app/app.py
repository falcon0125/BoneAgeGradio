import gradio as gr
import numpy as np
import pandas as pd
import json

import albumentations as A
from albumentations import ( 
    HorizontalFlip, CLAHE, GridDistortion, ShiftScaleRotate, Resize, RandomBrightnessContrast
)



transforms = A.Compose([
    ShiftScaleRotate(always_apply=False, p=1.0, shift_limit=(-0.05, 0.05), scale_limit=(-0.1, 0.2), rotate_limit=(-5, 5), interpolation=1, border_mode=0, value=(0, 0, 0), mask_value=None),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1),
    GridDistortion(always_apply=False, p=1.0, num_steps=6, distort_limit=(-0.2, 0.2), interpolation=1, border_mode=0, value=(0, 0, 0), mask_value=None),
    CLAHE(always_apply=False, p=0.5, clip_limit=(1, 4), tile_grid_size=(8, 8)),
    HorizontalFlip(p=0.5),
    Resize(p=1.0, height=256, width=256, interpolation=0),]
)
gr.set_static_paths(paths=["/app/atlas/"])

js = json.load(open("atlas.json",'r'))
"""[{"path":"atlas/girl-8m.jpg","name":"girl-8m","gender":"girl","ageo":"8m","age":0.6666666667},"""
delta = 3
img_height, img_width = 256,256
batch_size = 16
model = tf.keras.models.load_model(r"pruned_bone_age.h5", compile=False)  

def inference(gender, image_np ):
    sex = 1 if gender == 'boy' else 0    
    if len(image_np.shape)>2:
        image = image_np[:,:,1]
    else:
        image = image_np
    transform_f = lambda x: transforms(image=x)['image']
    img_list = ([image] * batch_size)
    imga = np.array(list(map(transform_f,img_list)))
    imga = imga / 256
    
    predict = model.predict([imga,np.array([sex]*batch_size)], batch_size=batch_size)
    s = pd.Series(predict.reshape(-1))
    argmax, argmin = s.argmax(), s.argmin()
    del s[argmax]
    del s[argmin]
    output = f"predict Bone Age: {s.mean():.1f} +/- {s.std():.1f} y/o\ngender:{gender}"
    df = s.to_frame()
    age_range = s.mean()
    print(output)
    #print(s)
    pathlist = [j['path'] for j in js if 
                j["age"]>(age_range-delta) and j["age"]<(age_range+delta) and j['gender']==gender]
    return output, gr.Gallery(pathlist, label="Atlas", show_download_button=False, show_share_button=False)

demo = gr.Interface(
    fn=inference,
    inputs=[ gr.Radio(["boy","girl"], label='Gender', value='boy'), gr.Image(label="Image")],
    outputs=[gr.Text(label="result"),gr.Gallery(label="Atlas")],
    title="Bone Age",
    description=r"Bone age AI training with 8K+ bone age data base on [Tanner-Whitehouse mathod](http://vl.academicdirect.ro/medical_informatics/bone_age/v1.0/)",
    article=r"Due to the fact that the population mainly consists of 6 to 10 year-olds, accuracy outside this range may be limited. <br/>Disclaimer: This information is intended for general purposes only and should not be used for medical purposes. Always consult a qualified healthcare professional for medical advice, diagnosis, or treatment",
    allow_flagging = 'manual',
    flagging_options=['agree','disagree'] 
)
#model = keras.layers.TFSMLayer(r"pruned_bone_age.pb", call_endpoint='serving_default')

if __name__ == "__main__":
    demo.launch(allowed_paths=["/app/atlas","/image"], favicon_path="hello.png",server_port=80)
