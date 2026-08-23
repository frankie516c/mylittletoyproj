# -*- coding: utf-8 -*-
"""임의 문장을 모든 모델에 던져보는 대화형 프로브.

  python probe.py "재미없지 않았어요" "그럭저럭 볼만했다"
  echo 문장 | python probe.py -
"""
import sys

import predictor as PR


def main():
    args = sys.argv[1:]
    if args == ["-"]:
        texts = [l.strip() for l in sys.stdin if l.strip()]
    else:
        texts = args
    if not texts:
        print(__doc__)
        return
    models = PR.load_all()
    for m in models:
        m.calibrate_polarity()
    w = max(len(t) for t in texts)
    print("문장".ljust(min(w, 50)) + "".join(f"{m.name[:14]:>16}" for m in models))
    for t in texts:
        row = t[:50].ljust(min(w, 50))
        for m in models:
            p = float(m.predict_proba([t])[0])
            row += f"{f'{p:.3f}({chr(43) if p>=0.5 else chr(45)})':>16}"
        print(row)


if __name__ == "__main__":
    main()
