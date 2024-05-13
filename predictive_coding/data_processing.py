import numpy as np
import os 

def remove_consecutive_repeats(np_file_path):
    """
    Returns an array of indices with the consecutive repeating frames removed. To be
    used on coordinate files and then applied to the frame files.
    """
    unique_idx = []
    data = np.load(np_file_path, allow_pickle=True)
    for i in range(len(data) - 1):
        if not np.array_equal(data[i], data[i + 1]):
            unique_idx.append(i)

    if len(data) > 0:
        unique_idx.append(len(data) - 1)
    del data
    return unique_idx
    
    
def map_files_to_chunks(source_directory, target_directory, file_start, seq_len, cont=False):
    print('Indexing files to', target_directory)

    #if not cont:
    os.makedirs(target_directory)
    file_index = 0
    #else:
    #    pattern = os.path.join(source_directory, '*.npy')
    #    files = sorted(glob.glob(pattern))
    #    print(files[-1][:-3])
    #    file_index = int(files[-1][:-3])

    for filename in os.listdir(source_directory):
        print(filename)
        if filename.startswith(file_start):
            filepath = os.path.join(source_directory, filename)
            data = np.load(filepath, mmap_mode='r')
            data = np.load(filepath, mmap_mode='r')
    
            for i in range(len(data) - (seq_len + 1)):
                chunk = data[i:i + seq_len + 1]
                chunk_fp = os.path.join(target_directory, f'{file_index}.npy')
                np.save(chunk_fp, chunk)
                file_index += 1
    
            del data
            #os.remove(filepath)

    return file_index

def map_dual_files_to_chunks(source_directory, target_directories, file_starts, seq_len, cont=False):
    print('Indexing files to', target_directories)

    os.makedirs(target_directories[0])
    os.makedirs(target_directories[1])
    file_index = 0

    for filename in os.listdir(source_directory):
        if filename.startswith(file_start[0]):
            filepath = os.path.join(source_directory, filename)
            data = np.load(filepath, mmap_mode='r')
    
            for i in range(len(data) - (seq_len + 1)):
                chunk = data[i:i + seq_len + 1]
                chunk_fp = os.path.join(target_directories[0], f'{file_index}.npy')
                np.save(chunk_fp, chunk)
                file_index += 1
    
            del data
            os.remove(filepath)
            
        elif filename.startswith(file_start[1]):
            filepath = os.path.join(source_directory, filename)
            data = np.load(filepath, mmap_mode='r')
    
            for i in range(len(data) - (seq_len + 1)):
                chunk = data[i:i + seq_len + 1]
                chunk_fp = os.path.join(target_directories[1], f'{file_index}.npy')
                np.save(chunk_fp, chunk)
                file_index += 1
    
            del data
            os.remove(filepath)

    return file_index
