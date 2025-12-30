#!/usr/bin/env zsh
set -e
cd "$(dirname "$0")"
ARG=${1:-arxiv.tex}
TARGET=$(basename "$ARG")
LATEXMK=$(command -v latexmk || true)
XELATEX=$(command -v xelatex || true)
if [[ -z "$LATEXMK" && -x /Library/TeX/texbin/latexmk ]]; then LATEXMK=/Library/TeX/texbin/latexmk; fi
if [[ -z "$XELATEX" && -x /Library/TeX/texbin/xelatex ]]; then XELATEX=/Library/TeX/texbin/xelatex; fi
if command -v tectonic >/dev/null 2>&1; then
  tectonic "$TARGET"
elif [[ -n "$XELATEX" ]]; then
  "$XELATEX" -interaction=nonstopmode "$TARGET"
  "$XELATEX" -interaction=nonstopmode "$TARGET"
elif [[ -n "$LATEXMK" ]]; then
  if [[ -n "$XELATEX" ]]; then
    "$LATEXMK" -e "$xelatex=q{$XELATEX %O %S}" -xelatex "$TARGET"
  else
    "$LATEXMK" -xelatex "$TARGET"
  fi
else
  echo "未检测到 LaTeX 编译器：请安装 Tectonic 或 MacTeX 后再运行"
  exit 1
fi
