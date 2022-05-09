import towhee

def trans_img(input_img, version):
    if version == 'celeba':
        x = towhee.ops.img2img_translation.animegan(model_name = 'celeba')(input_img)
    elif version == 'facepaintv1':
        x = towhee.ops.img2img_translation.animegan(model_name = 'facepaintv1')(input_img)
    elif version == 'facepaintv2':
        x = towhee.ops.img2img_translation.animegan(model_name = 'facepaintv2')(input_img)
    elif version == 'hayao':
        x = towhee.ops.img2img_translation.animegan(model_name = 'hayao')(input_img)
    elif version == 'paprika':
        x = towhee.ops.img2img_translation.animegan(model_name = 'paprika')(input_img)
    elif version == 'shinkai':
        x = towhee.ops.img2img_translation.animegan(model_name = 'shinkai')(input_img)

    return x.cv2_to_rgb()

import gradio

interface = gradio.Interface(trans_img, 
                             [gradio.inputs.Image(type="pil", source='upload'),
                              gradio.inputs.Radio(["celeba", "facepaintv1","facepaintv2", 
                                                  "hayao", "paprika", 'shinkai'])],
                             gradio.outputs.Image(type="numpy"), allow_flagging='never', allow_screenshot=False)

interface.launch(enable_queue=True)

# %%



