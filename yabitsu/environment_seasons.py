"""広葉樹だけを秋色へ変えるCSP条件。時期・強さは景観用の設計値。"""
from . import ext_config


def write_config(track_dir):
    ext_config.write_part(track_dir, 'seasons', '''; CSP Trees seasonal conditions
; https://github.com/ac-custom-shaders-patch/acc-extension-config/wiki/Tracks-%E2%80%93-Trees
; 冬の落葉は含まない。杉・檜・下草は色を変えない。
[CONDITION_...]
NAME=YABITSU_AUTUMN
INPUT=YEAR_PROGRESS
LUT=(|0=0.5|0.15=0.15|0.2=0|0.73=0|0.79=0.1|0.84=0.8|0.88=1|0.95=0.75|1=0.5|)

[TREES]
SEASON_AUTUMN_0=YABITSU_AUTUMN
SEASON_AUTUMN_1=sugi?,hinoki?,shrub?,fern?,0
''')
