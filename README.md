# upgraded-pysearcher

tired of slow search tools on windows and linux? this uses cpu + cuda gpu to make searching insanely fast ⚡

how’d i build it? asked chatgpt lol. this is my first legit finished project with it, so yeah, hype.

---

## Features

- full CLI support (no GUI, because nah)  
- tons of customizable flags  
- live thread status updates  
- file extension filtering  
- verbose output mode  
- fake scan mode (just for kicks)  
- scan entire disks or multiple disks at once  
- **speedrun mode**: max threads, no safety checks, pure speed 🚀  
- skip common system folders (windows only, linux maybe later)  
- detailed error reporting  
- built-in help panel  
- advanced content filters: search by hex, base64, or file size  

---

if you want gui or more features, hit me up. otherwise, happy blazing through files 🔥

---
#### TODO
>- full cupy support
>- use actual tensors and not imitat-ey bs
>- gui, probably not since i hate it
>- add opencl support for non-nvidia users (will be cuda only on green team)
>- upgrade the rich presence because it looks a$$
>- linux support (YES YOU USE ARCH BTW)
>- make 2mb ram caches
>- ditch the need for cpu power
>- reduce useless load/cycle through folders (more of that)
>- auto update from release
>- implement ai to anger Jensen
>- add antivirus-based skips (will be a flag, -av off/on)
>- alleviate load for "portable people" when low battery
>- trigger a warning saying "slow as $h1t" if r/w is slower than 100mbps
>- anti-idle, what are we, regular people?
>- useful live perf counters
