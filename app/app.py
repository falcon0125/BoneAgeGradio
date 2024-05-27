import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
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

img_height, img_width = 256,256
batch_size = 32
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
    output = f"predict Bone Age: {s.mean():.1f} +/- {s.std():.1f} y/o\nmale:{gender}"
    df = s.to_frame()
    print(output)
    #print(s)
    return output

demo = gr.Interface(
    fn=inference,
    inputs=[ gr.Radio(["boy","girl"]), gr.Image()],
    outputs=["text"],
)
#model = keras.layers.TFSMLayer(r"pruned_bone_age.pb", call_endpoint='serving_default')

if __name__ == "__main__":
    demo.launch()
