import gradio as gr

#Holds all the inputs
input_list = []

# maxitem is the max number of visible inputs
max_item = 9
# lastidx is the index to begin the offset from
def change_visibility(inputType, last_idx):
    if inputType == 'clear':
        return [0] + [gr.update(visible=False)] * max_item * 3
    elif inputType == 'text':
        offset = 0
    elif inputType == 'img':
        offset = 1
    elif inputType == 'aud':
        offset = 2

    if last_idx < max_item * 3:
        #Updates nothing for previous inputs, updates the new input to visible, and updates nothing for the rest of the inputs
        return [last_idx + 3] + [gr.update()] * (last_idx + offset) + [gr.update(visible=True, interactive=True)] * (1) + [gr.update()] * (3 * max_item - last_idx - offset - 1)
    else:
        return [last_idx] + [gr.Textbox.update()] * max_item * 3

customCSS = """
.grid-container.svelte-1b19cri.svelte-1b19cri {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

.thumbnail-lg.svelte-1b19cri.svelte-1b19cri{
    width: unset;
    height: 35vh;
    aspect-ratio: auto;
}
"""
with gr.Blocks(css=customCSS) as demo:
    gr.Markdown( f"# Add your inputs here. A maxiumum of {max_item} inputs can be added.")
    with gr.Row():
        last_idx = gr.State(value=0)
        textState = gr.State(value='text')
        imgState = gr.State(value='img')
        audState = gr.State(value='aud')
        clearState = gr.State(value='clear')
        for i in range(max_item):
            txt = gr.Textbox(visible=False, placeholder="Input text here...")
            input_list.append(txt)
            img = gr.Image(visible=False)
            input_list.append(img)
            aud = gr.Audio(visible=False)
            input_list.append(aud)
    with gr.Row():
        addtxt = gr.Button("Add textbox")
        addimg = gr.Button("Add image")
        addaud = gr.Button("Add audio")
        clearAll = gr.ClearButton(value="Clear all inputs", components=input_list)
        addtxt.click(change_visibility, [textState, last_idx], [
                        last_idx, *input_list])
        addimg.click(change_visibility, [imgState, last_idx], [
                        last_idx, *input_list])
        addaud.click(change_visibility, [audState, last_idx], [
                        last_idx, *input_list])
        clearAll.click(change_visibility, [clearState, last_idx], [
                        last_idx, *input_list])
            #links to 10 random images with different sizes
    image_urls = ["https://cdn.pixabay.com/photo/2023/06/27/03/21/lizard-8091302_1280.jpg", "https://cdn.pixabay.com/photo/2023/08/13/00/43/blue-8186653_1280.jpg",
                  "https://cdn.pixabay.com/photo/2023/07/05/18/13/mountains-8108961_1280.jpg", "https://cdn.pixabay.com/photo/2023/08/02/14/25/dog-8165447_1280.jpg",
                  "https://cdn.pixabay.com/photo/2023/08/11/10/13/goat-8183257_1280.png",
                  "https://cdn.pixabay.com/photo/2023/08/12/04/24/mabry-mill-8184715_1280.jpg"]
    outputGallery = gr.Gallery(value=image_urls, label="Gallery")



demo.launch()
