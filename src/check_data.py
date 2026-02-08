
import datasets
from datasets import load_from_disk
import os

def check_data():
    dataset_path = "datasets/washing_machine_corpus"
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return

    ds = load_from_disk(dataset_path)
    print(f"Dataset loaded. Type: {type(ds)}")
    
    data_list = []
    if hasattr(ds, 'keys'):
        print(f"Splits: {ds.keys()}")
        for split in ds.keys():
            data_list.extend(ds[split])
    else:
        data_list = ds

    print(f"Total samples: {len(data_list)}")
    
    # Filter
    wm_count = 0
    w_count = 0
    m_count = 0
    
    wm_examples = []
    w_examples = []
    m_examples = []

    for item in data_list:
        text = item['text'].lower()
        if "washing machine" in text:
            wm_count += 1
            if len(wm_examples) < 3: wm_examples.append(text)
        elif "washing" in text:
            w_count += 1
            if len(w_examples) < 3: w_examples.append(text)
        elif "machine" in text:
            m_count += 1
            if len(m_examples) < 3: m_examples.append(text)

    print(f"Washing Machine count: {wm_count}")
    print(f"Washing Only count: {w_count}")
    print(f"Machine Only count: {m_count}")
    
    print("\n--- Washing Machine Examples ---")
    for ex in wm_examples: print(ex[:100] + "...")
    
    print("\n--- Washing Only Examples ---")
    for ex in w_examples: print(ex[:100] + "...")
    
    # Deep dive into "washing" contexts
    print("\n--- Investigating 'Washing' Contexts ---")
    found_compound = False
    for item in data_list:
        text = item['text'].lower()
        if "washing" in text:
            indices = [i for i in range(len(text)) if text.startswith("washing", i)]
            for i in indices:
                context = text[i:i+30]
                # print(f"Context: '{context}'") 
                if "machine" in context:
                    print(f"FOUND NEARBY: '{context}'")
                    found_compound = True
    
    if not found_compound:
        print("No instances of 'washing' followed closely by 'machine' found.")
