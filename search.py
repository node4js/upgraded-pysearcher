import os
import sys
import threading
import queue
import base64
import time
import argparse
import psutil
import pynvml
from rich.live import Live
from rich.table import Table
from rich.console import Console

from numba import cuda
import numpy as np

def check_system_limits():
    max_ram_gb = 32
    max_vram_gb = 8

    total_ram = psutil.virtual_memory().total / (1024 ** 3)
    if total_ram > max_ram_gb:
        print(f"[WARN] {total_ram:.1f} GB RAM - Not supported")
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram = mem_info.total / (1024 ** 3)
        if total_vram > max_vram_gb:
            print(f"[WARN] {total_vram:.1f} GB VRAM - Not supported")
        pynvml.nvmlShutdown()
    except:
        print("[INFO] Couldn’t check VRAM (no supported GPU or nvml issue)")


def main():
    check_system_limits()  
    args = parse_args()

def chunk_reader(fobj, chunk_size=2*1024*1024):
    while True:
        data = fobj.read(chunk_size)
        if not data:
            break
        yield data

def is_system_folder(path):
    sys_folders = ['AppData', 'Local', 'Microsoft', 'Windows', 'ProgramData', 'Program Files', 'Program Files (x86)']
    return any(part in path.split(os.sep) for part in sys_folders)

@cuda.jit
def cuda_search_kernel(data, pattern, result):
    i = cuda.grid(1)
    n = data.size
    m = pattern.size
    if i + m <= n:
        match = True
        for j in range(m):
            if data[i + j] != pattern[j]:
                match = False
                break
        if match:
            result[0] = 1

def cuda_search(data_bytes, pattern_bytes):
    data_np = np.frombuffer(data_bytes, dtype=np.uint8)
    pattern_np = np.frombuffer(pattern_bytes, dtype=np.uint8)
    result = np.zeros(1, dtype=np.uint8)
    threads_per_block = 256
    blocks = (len(data_np) + threads_per_block - 1) // threads_per_block
    d_data = cuda.to_device(data_np)
    d_pattern = cuda.to_device(pattern_np)
    d_result = cuda.to_device(result)
    cuda_search_kernel[blocks, threads_per_block](d_data, d_pattern, d_result)
    d_result.copy_to_host(result)
    return result[0] == 1

class SearchThread(threading.Thread):
    def __init__(self, tid, task_queue, results, args, status_dict, counter_lock, files_scanned):
        super().__init__()
        self.tid = tid
        self.task_queue = task_queue
        self.results = results
        self.args = args
        self.status_dict = status_dict
        self.daemon = True
        self.counter_lock = counter_lock
        self.files_scanned = files_scanned
        self.cuda_available = False
        try:
            cuda.select_device(0)
            self.cuda_available = True
        except cuda.cudadrv.error.CudaSupportError:
            self.cuda_available = False

    def run(self):
        while True:
            try:
                path = self.task_queue.get(timeout=1)
            except queue.Empty:
                self.status_dict[self.tid] = "> Idle"
                break

            self.status_dict[self.tid] = f"[INFO] Scanning {os.path.basename(path)}"

            if self.args.skip_sys and is_system_folder(path):
                self.status_dict[self.tid] = "> Idle"
                self.task_queue.task_done()
                continue

            if os.path.isdir(path):
                if not self.args.nosub:
                    try:
                        for entry in os.listdir(path):
                            full_path = os.path.join(path, entry)
                            self.task_queue.put(full_path)
                    except Exception as e:
                        if self.args.errors:
                            print(f"[ERROR] Cannot list {path}: {e}")
            else:
                filename = os.path.basename(path)

                if self.args.fc and filename != self.args.fc:
                    self.status_dict[self.tid] = "> Idle"
                    self.task_queue.task_done()
                    continue

                if self.args.ext and not filename.lower().endswith(self.args.ext.lower()):
                    self.status_dict[self.tid] = "> Idle"
                    self.task_queue.task_done()
                    continue

                try:
                    fsize = os.path.getsize(path)
                    if self.args.minsize and fsize < self.args.minsize:
                        self.status_dict[self.tid] = "> Idle"
                        self.task_queue.task_done()
                        continue
                    if self.args.maxsize and fsize > self.args.maxsize:
                        self.status_dict[self.tid] = "> Idle"
                        self.task_queue.task_done()
                        continue

                    if self.args.amongus and fsize == 69420:
                        print(f"[?] Sus file found: {path}")

                except Exception as e:
                    if self.args.errors:
                        print(f"[ERROR] cannot get size {path}: {e}")
                    self.status_dict[self.tid] = "> Idle"
                    self.task_queue.task_done()
                    continue

                found_content = False
                try:
                    with open(path, 'rb') as f:
                        for chunk in chunk_reader(f):
                            if self.args.fake:
                                found_content = True
                                break

                            if self.args.content:
                                search_bytes = self.args.content.encode(errors='ignore')

                                if self.cuda_available:
                                    if cuda_search(chunk, search_bytes):
                                        found_content = True
                                        break
                                else:
                                    if search_bytes.lower() in chunk.lower():
                                        found_content = True
                                        break

                            elif self.args.hex:
                                try:
                                    target_bytes = bytes.fromhex(self.args.hex)
                                    if self.cuda_available:
                                        if cuda_search(chunk, target_bytes):
                                            found_content = True
                                            break
                                    else:
                                        if target_bytes in chunk:
                                            found_content = True
                                            break
                                except Exception:
                                    pass
                            elif self.args.base64:
                                try:
                                    target_bytes = base64.b64decode(self.args.base64)
                                    if self.cuda_available:
                                        if cuda_search(chunk, target_bytes):
                                            found_content = True
                                            break
                                    else:
                                        if target_bytes in chunk:
                                            found_content = True
                                            break
                                except Exception:
                                    pass

                    if self.args.fc:
                        if found_content or not self.args.content:
                            self.results.append(path)
                            if not self.args.quiet:
                                print(f"[MATCH] {path} (filename match)")
                    else:
                        if self.args.content and found_content:
                            self.results.append(path)
                            if not self.args.quiet:
                                print(f"[MATCH] {path} (contains '{self.args.content}')")
                        elif not self.args.content and not self.args.fc:
                            self.results.append(path)
                            if not self.args.quiet and self.args.noisy:
                                print(f"[FILE] {path}")
                except Exception as e:
                    if self.args.errors:
                        print(f"[ERROR] Cannot open/read {path}: {e}")

                with self.counter_lock:
                    self.files_scanned[0] += 1

                self.status_dict[self.tid] = "> Idle"
            self.task_queue.task_done()
            if self.args.delay:
                time.sleep(self.args.delay)

def parse_args():
    parser = argparse.ArgumentParser(description="Epic Multithreaded Python Search Tool (arguments are stackable)")

    parser.add_argument("path", nargs="?", default="C:\\", help="Path or disk(s) to search")

    parser.add_argument("-adisks", action="store_true", help="Search all available fixed disks")
    parser.add_argument("-spdisk", type=str, help="Specify one disk (like C:\\)")
    parser.add_argument("-wdisk", action="store_true", help="Search root of disks instead of folders")

    parser.add_argument("-nosub", action="store_true", help="No recursion into subdirectories")
    parser.add_argument("-fc", type=str, help="Focus on specific filename only")
    parser.add_argument("-content", type=str, help="Search inside files for text")
    parser.add_argument("-hex", type=str, help="Search for hex-encoded bytes inside files")
    parser.add_argument("-base64", type=str, help="Search for base64-encoded string inside files")
    parser.add_argument("-ext", type=str, help="Only scan files with this extension")

    parser.add_argument("-minsize", type=int, help="Only scan files bigger than this size (bytes)")
    parser.add_argument("-maxsize", type=int, help="Only scan files smaller than this size (bytes)")

    parser.add_argument("-threads", type=int, help="Number of threads to use")
    parser.add_argument("-noisy", action="store_true", help="Print every file scanned")
    parser.add_argument("-quiet", action="store_true", help="Suppress status except matches")
    parser.add_argument("-nocache", action="store_true", help="Disable caching chunks to temp files (ignored)")
    parser.add_argument("-keeptemp", action="store_true", help="Do NOT delete temp cache files (ignored)")
    parser.add_argument("-stats", action="store_true", help="Show final stats")
    parser.add_argument("-fake", action="store_true", help="Simulate scanning, no real file open")
    parser.add_argument("-delay", type=float, help="Sleep seconds between scans")
    parser.add_argument("-errors", action="store_true", help="Show detailed errors")
    parser.add_argument("-skip_sys", action="store_true", help="Skip common system folders (C:\\Users only)")

    parser.add_argument("-skibidibounce", action="store_true", help="Random delays + glitchy output")
    parser.add_argument("-speedrun", action="store_true", help="Max threads + no safety checks")
    parser.add_argument("-amongus", action="store_true", help="Say 'sus file' on file size 69420")
    parser.add_argument("-nuke", action="store_true", help="Scary name, does nothing lol")
    parser.add_argument("-😎", action="store_true", help="Cool output mode (animated?)")

    return parser.parse_args()

def build_status_table(status_dict, files_scanned, elapsed, args):
    table = Table(title="Thread Statuses (live)")

    table.add_column("Thread ID", justify="center", style="white", no_wrap=True)
    table.add_column("Status", style="black")

    for tid, status in sorted(status_dict.items()):
        table.add_row(str(tid), status)

    progress_str = f"Files scanned: {files_scanned[0]}"
    if elapsed > 0 and files_scanned[0] > 0:
        rate = files_scanned[0] / elapsed
        progress_str += f" | Rate: {rate:.1f} files/s"
    else:
        rate = 0

    if args.speedrun:
        progress_str += "Speedrun mode ON"

    table.caption = progress_str
    return table

def main():
    args = parse_args()

    if args.speedrun:
        args.threads = 32
        args.nocache = True
        args.quiet = True
        args.errors = False

    start_time = time.time()
    task_queue = queue.Queue()
    results = []
    status_dict = {}
    counter_lock = threading.Lock()
    files_scanned = [0]  

    base_path = args.spdisk or args.path
    if args.adisks:
        for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
            drive = f"{letter}:\\" 
            if os.path.exists(drive):
                task_queue.put(drive)
    else:
        task_queue.put(base_path)

    thread_count = args.threads or os.cpu_count()
    threads = []

    for i in range(thread_count):
        status_dict[i] = "init"
        t = SearchThread(i, task_queue, results, args, status_dict, counter_lock, files_scanned)
        threads.append(t)
        t.start()

    console = Console()
    try:
        with Live(build_status_table(status_dict, files_scanned, 0, args), console=console, refresh_per_second=5) as live:
            while any(t.is_alive() for t in threads):
                elapsed = time.time() - start_time
                live.update(build_status_table(status_dict, files_scanned, elapsed, args))
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[!] CTRL+c detected. Exiting gracefully...")
        sys.exit(1)

    if args.stats:
        elapsed = time.time() - start_time
        print(f"\n\n[STATS] Done in {elapsed:.2f}s. Files matched: {len(results)} | Scanned: {files_scanned[0]}")

if __name__ == "__main__":
    main()