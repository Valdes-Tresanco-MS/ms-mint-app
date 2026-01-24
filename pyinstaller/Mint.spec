# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

src_dir = os.path.abspath(os.path.join(SPECPATH, os.pardir))
hooks_dir = os.path.join(src_dir, 'pyinstaller', 'hooks')
package_root = os.path.join(src_dir, 'src')
script = os.path.join(package_root, 'ms_mint_app', 'scripts', 'Mint.py')


def _safe_collect_submodules(module_name: str):
    try:
        return collect_submodules(module_name)
    except Exception:
        return []


all_hidden_imports = (
    _safe_collect_submodules('sklearn')
    + _safe_collect_submodules('bs4')
    + _safe_collect_submodules('scipy')
    + _safe_collect_submodules('pyarrow')
    + _safe_collect_submodules('ms_mint_app')
    + _safe_collect_submodules('packaging')
    + _safe_collect_submodules('brotli')
    + _safe_collect_submodules('waitress')
    + _safe_collect_submodules('dash')
    + _safe_collect_submodules('dash_extensions')
    + _safe_collect_submodules('feffery_antd_components')
    + _safe_collect_submodules('feffery_utils_components')
    + _safe_collect_submodules('numpy')
    + _safe_collect_submodules('webview')
    + _safe_collect_submodules('diskcache')
    + _safe_collect_submodules('flask_caching')
    + _safe_collect_submodules('duckdb')
    + _safe_collect_submodules('polars')
    + _safe_collect_submodules('fastcluster')
)


import sys

# Path to the bundled Asari environment (created by create_asari_env.py)
asari_env_dir = os.path.join(SPECPATH, 'asari_env')

# Platform-specific configurations
binaries_list = []
datas_list = [
    (os.path.join(package_root, 'ms_mint_app', 'assets'), os.path.join('ms_mint_app', 'assets')),
    (os.path.join(package_root, 'ms_mint_app', 'static'), os.path.join('ms_mint_app', 'static')),
]

# Add Asari environment if it exists
if os.path.isdir(asari_env_dir):
    datas_list.append((asari_env_dir, 'asari_env'))

# ============================================================================
# MATPLOTLIB FONT CACHE - Startup Optimization
# ============================================================================
# Bundle pre-built matplotlib font cache to eliminate 5-second first-time delay.
# The cache is platform-specific due to different system fonts.
# 
# To rebuild cache (when matplotlib is updated):
#   cd pyinstaller
#   python prebuild_matplotlib_cache.py
#
# This copies the prebuild_matplotlib_cache/matplotlib/ directory into assets/
# during the build process.
# ============================================================================

import platform
matplotlib_cache_src = os.path.join(SPECPATH, 'prebuild_matplotlib_cache', 'matplotlib')

# Check if pre-built cache exists
if os.path.isdir(matplotlib_cache_src):
    # Bundle with platform identifier for cross-platform builds
    system = platform.system().lower()
    datas_list.append((matplotlib_cache_src, os.path.join('assets', f'matplotlib_cache_{system}')))
    print(f"✓ Bundling matplotlib cache for {system} ({os.path.getsize(os.path.join(matplotlib_cache_src, 'fontlist-v390.json'))/1024:.1f} KB)")
else:
    print("⚠️  Matplotlib cache not found. Run 'python prebuild_matplotlib_cache.py' in pyinstaller/ to generate it.")
    print("   Without cache, first-time users will experience ~5s startup delay.")


# Platform-specific: PyOpenMS share directory and binaries
if sys.platform == 'linux':
    # PyOpenMS share directory path for Linux
    pyopenms_share = os.path.join(
        sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages', 'pyopenms', 'share'
    )
    if os.path.isdir(pyopenms_share):
        datas_list.append((pyopenms_share, 'pyopenms_share'))
    
    # Linux-specific binaries to fix library version conflicts
    linux_libs = [
        ('libssl.so.3', '.'),
        ('libcrypto.so.3', '.'),
        ('libstdc++.so.6', '.'),
    ]
    for lib_name, dest in linux_libs:
        lib_path = os.path.join(sys.prefix, 'lib', lib_name)
        if os.path.exists(lib_path):
            binaries_list.append((lib_path, dest))

elif sys.platform == 'darwin':  # macOS
    # PyOpenMS share directory path for macOS
    pyopenms_share = os.path.join(
        sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages', 'pyopenms', 'share'
    )
    if os.path.isdir(pyopenms_share):
        datas_list.append((pyopenms_share, 'pyopenms_share'))

elif sys.platform == 'win32':  # Windows
    # PyOpenMS share directory path for Windows
    pyopenms_share = os.path.join(
        sys.prefix, 'Lib', 'site-packages', 'pyopenms', 'share'
    )
    if os.path.isdir(pyopenms_share):
        datas_list.append((pyopenms_share, 'pyopenms_share'))

# Runtime hooks - only include OpenMS hook if pyopenms_share was bundled
runtime_hooks_list = []
if any('pyopenms_share' in str(d) for d in datas_list):
    runtime_hooks_list.append(os.path.join(hooks_dir, 'rthook_openms.py'))

a = Analysis(
    [script],
    pathex=[src_dir, package_root],
    hookspath=[hooks_dir],
    hiddenimports=all_hidden_imports,
    module_collection_mode={
        'ms_mint_app': 'pyz+py',
    },
    binaries=binaries_list,
    datas=datas_list,
    excludes=['PySide6'],  # Exclude to avoid Qt bindings conflict with PyQt5
    runtime_hooks=runtime_hooks_list,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Mint',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
