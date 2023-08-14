import gradio as gr

input_list = []

#lastidx starts from 1
def change_visibility(inputType, last_idx):
    if inputType == 'text':
        offset = 1
    elif inputType == 'img':
        offset = 2
    elif inputType == 'aud':
        offset = 3
    
    update_idx = last_idx + offset
    
    if last_idx < 50:
        return [last_idx + 3] + [gr.update()] * (last_idx + offset - 1) + [gr.update(visible=True, interactive = True)] * (1) + [gr.update()] * (17)
    else:
        return [num_item] + [gr.Textbox.update(visible=True)] * num_item
    

with gr.Blocks() as demo:
    gr.Markdown("Add your inputs here.")
    with gr.Row():
        last_idx = gr.State(value = 0)
        textState = gr.State(value = 'text')
        imgState = gr.State(value = 'img')
        audState = gr.State(value = 'aud')
        for i in range(6):
            txt = gr.Textbox(visible=False)
            input_list.append(txt)
            img = gr.Image(visible=False)
            input_list.append(img)
            aud = gr.Audio(visible=False)
            input_list.append(aud)
        with gr.Column():
            addtxt = gr.Button("Add textbox")
            addimg = gr.Button("Add image")
            addaud = gr.Button("Add audio")
            addtxt.click(change_visibility, [textState, last_idx], [last_idx, *input_list])
            addimg.click(change_visibility, [imgState, last_idx], [last_idx, *input_list])
            addaud.click(change_visibility, [audState, last_idx], [last_idx, *input_list])
    
demo.launch()


# import gradio as gr

# input_list = [[],[],[]]

# def change_visibility(num_item):
#     if num_item < 7:
#         return [num_item + 1] + [gr.update(visible=True)] * (num_item + 1) + [gr.update(visible=False)] * (2 - num_item)
#     else:
#         return [num_item] + [gr.Textbox.update(visible=True)] * num_item
    

# with gr.Blocks() as demo:
#     gr.Markdown("Add your inputs here.")
#     with gr.Row():
#         num_txt = gr.State(value = 0)
#         num_img = gr.State(value = 0)
#         num_aud = gr.State(value = 0)
#         for i in range(3):
#             txt = gr.Textbox(visible=False)
#             input_list[0].append(txt)
#             img = gr.Image(visible=False)
#             input_list[1].append(img)
#             aud = gr.Audio(visible=False)
#             input_list[2].append(aud)
#         with gr.Column():
#             addtxt = gr.Button("Add textbox")
#             addimg = gr.Button("Add image")
#             addaud = gr.Button("Add audio")
#             addtxt.click(change_visibility, num_txt, [num_txt, *input_list[0]])
#             addimg.click(change_visibility, num_img, [num_img, *input_list[1]])
#             addaud.click(change_visibility, num_aud, [num_aud, *input_list[2]])
    
# demo.launch()