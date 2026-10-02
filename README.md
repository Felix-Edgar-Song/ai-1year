[![Live Demo](https://img.shields.io/badge/Live%20Demo-12MiB%20P99%201.1ms%2050x%2012.38GiB-brightgreen)](https://vagrantly-creatable-stem.ngrok-free.dev/vram) [![GitHub](https://img.shields.io/badge/GitHub-Felix--Edgar--Song%2Fblackwell--llm--serving-blue)](https://github.com/Felix-Edgar-Song/blackwell-llm-serving)

Live Demo: https://vagrantly-creatable-stem.ngrok-free.dev/vram | Health: /health | Generate: /generate?prompt=Hello%20Meta%20SG - RTX 5060 Ti 16GB sm_120 Blackwell 36 SMs 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.385GiB<16GiB Least VRAM LB 8000-8049

## 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 100 runs 50 replicas 12.385GiB<16GiB Least VRAM LB 8000-8049 /vram health /generate 5060 Ti 16GB sm_120 Blackwell 36 SMs 74.2% cut 902MiB 17→232.7MiB 50 3x 2-month

Logs: Day24 902MiB baseline → Day27 233.4MiB smashed 400MiB → Day30 233.4MiB 12.42GiB OK 74% 3x → Day32 232.7MiB 265.5MiB Peak P50 0.9ms P99 1.1ms 100 runs lowest+P99 → Day33 232.68MiB/265.45MiB FastAPI /vram /generate 7.45ms→1.1ms → Day34 232.7x3=743MiB OK x50=12.385GiB OK P99 1.1ms x50=55ms<100ms SLO nginx least_conn LB → Day35 3-Track 50 + Resume 1-line + Interview 4 + 12MiB P8 4W
