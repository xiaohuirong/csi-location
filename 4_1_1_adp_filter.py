import numpy as np
from utils.parse_args import parse_args, show_args
import tqdm
import multiprocessing as mp


args = parse_args()
show_args(args)

r = args.round
s = args.scene
m = args.method
p = args.port
o = args.over

f = args.filter

adp_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Adp{r}Scene{s}.npy"

adpf_path = f"data/round{r}/s{s}/feature/Port{p}Over{o}Filter{f}Adp{r}Scene{s}.npy"


adp = np.load(adp_path)


bsz = adp.shape[0]

adp_f = np.zeros((bsz, bsz), dtype=np.float32)

all_indices = np.argsort(adp, axis=-1)


def adp_filter(todo_queue, output_queue):
    def adp_dissimilarities(index):
        w = adp[index, :]
        r = np.zeros(bsz)
        for ii in range(bsz):
            indices = all_indices[ii, :1000]
            r[ii] = np.mean(w[indices])

        return r

    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        output_queue.put((index, adp_dissimilarities(index)))


with tqdm.tqdm(total=adp.shape[0]) as pbar:
    todo_queue = mp.Queue()
    output_queue = mp.Queue()

    for i in range(adp.shape[0]):
        todo_queue.put(i)

    thread_num = 12
    for i in range(thread_num):
        todo_queue.put(-1)
        p = mp.Process(target=adp_filter, args=(todo_queue, output_queue))
        p.start()

    finished_processes = 0
    while finished_processes != thread_num:
        i, d = output_queue.get()

        if i == -1:
            finished_processes = finished_processes + 1
        else:
            adp_f[i, :] = d
            pbar.update(1)
np.save(adpf_path, adp_f)
