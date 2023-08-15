import pyarrow as pa
from PIL import Image
import base64
import io
path = "/home/azureuser/coco_caption_karpathy_val.arrow"

# Open the Arrow file for reading
with pa.ipc.open_file(path) as reader:
    # Access the Arrow Table from the file
    table = reader.read_all()

# Convert the Arrow Table to a Pandas DataFrame (optional)
import pandas as pd
df = table.to_pandas()

# Now you can work with the DataFrame
binary_data = df['image'][0]

print(binary_data)

def load_pretrain_arrows(root, arrow_paths):

    """

    Args:

        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".

    Returns:

        list[dict]: a list of dicts in Detectron2 standard format. (See

        `Using Custom Datasets </tutorials/datasets.html>`_ )

    """

    arrs = []

    for arrow_path in arrow_paths:

        arr = pa.ipc.RecordBatchFileReader(

                        pa.memory_map(os.path.join(root, arrow_path), "r")

                    ).read_all()

        arrs.append(arr)

    return arrs

 

data_list = ["coco_caption_karpathy_train.arrow", "coco_caption_karpathy_val.arrow", "coco_caption_karpathy_test.arrow", "coco_caption_karpathy_restval.arrow"]

pretrain_arrows = load_pretrain_arrows("/mnt/vig_data/zhgan/data/ziyi_arrow/data/pretrain_arrows_code224", data_list)