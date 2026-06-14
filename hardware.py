"""Detección de hardware y recomendación de modelo según la gama del equipo.

El pipeline es agnóstico al modelo, así que el usuario puede elegir el motor de
traducción según su PC.  Este módulo estima RAM/CPU/GPU y recomienda el modelo
más inteligente que su máquina puede correr con holgura, ANTES de descargarlo.
"""
import os
import shutil
import subprocess
import sys


def _total_ram_gb() -> float:
    """RAM física total en GB, sin depender de psutil."""
    # psutil si está disponible (más fiable)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MS()
            stat.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
        except Exception:
            pass
    else:
        try:  # Linux
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024 ** 3)
        except Exception:
            pass
    return 8.0  # supuesto conservador si todo falla


def _has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def detect() -> dict:
    """Devuelve {ram_gb, cores, threads, gpu} del equipo actual."""
    return {
        "ram_gb": round(_total_ram_gb(), 1),
        "cores": os.cpu_count() or 2,
        "gpu": _has_nvidia_gpu(),
    }


def recommend(specs: dict | None = None) -> str:
    """ID del modelo recomendado para este equipo.

    Regla: el modelo más capaz cuyo `min_ram_gb` quepa dejando ~30 % de RAM
    libre para el SO y PyMuPDF.  Una GPU NVIDIA sube un escalón.
    """
    if specs is None:
        specs = detect()
    import translator
    usable = specs["ram_gb"] * 0.7
    # Ordenar de más capaz a menos
    by_quality = sorted(
        translator.MODELS.values(), key=lambda m: m["quality_rank"], reverse=True
    )
    for m in by_quality:
        need = m["min_ram_gb"]
        if specs["gpu"]:
            need *= 0.6  # con GPU el coste de RAM importa menos
        if need <= usable:
            return m["id"]
    # Si nada cabe, el más liviano
    return min(translator.MODELS.values(), key=lambda m: m["min_ram_gb"])["id"]


def tier_label(specs: dict | None = None) -> str:
    if specs is None:
        specs = detect()
    ram = specs["ram_gb"]
    if specs["gpu"] and ram >= 16:
        return "Gama alta (GPU)"
    if ram >= 16:
        return "Gama alta"
    if ram >= 8:
        return "Gama media"
    return "Gama básica"


if __name__ == "__main__":
    s = detect()
    print(f"Equipo: {s['ram_gb']} GB RAM · {s['cores']} hilos · "
          f"GPU NVIDIA: {'sí' if s['gpu'] else 'no'}")
    print(f"Gama: {tier_label(s)}")
    print(f"Modelo recomendado: {recommend(s)}")
