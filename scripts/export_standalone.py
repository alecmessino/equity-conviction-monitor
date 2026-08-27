"""Build a single self-contained HTML file: the terminal with its ledger baked in.

The deployed terminal fetches ten ledger artifacts over HTTP. That is right for a hosted
page and useless for one that has to travel — opened from a file:// URL every fetch fails
CORS, the board never loads, and the page renders its "could not load index.json" banner.

This inlines the ledger as a JSON blob and installs a tiny shim ahead of the app that
answers those fetches from memory. Nothing in web/terminal.html changes: the shim
intercepts `fetch`, so the app keeps its normal load path and there is no second code
path to keep in sync — the export cannot silently diverge from the page it is exporting.

Per-symbol files (ledger/history/SYM.json, ledger/factors/SYM.json) are lazily fetched by
the tear sheet and are gitignored, so they are inlined only when present. When they are
not, the shim returns a 404 exactly as the network would, and the panel degrades the way
it already knows how to.

    python scripts/export_standalone.py [-o OUTPUT]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERMINAL = os.path.join(ROOT, "web", "terminal.html")
LEDGER = os.path.join(ROOT, "ledger")

# The artifacts boot() fetches, in the order it fetches them. index.json is required;
# every other one already sits behind a bare try/catch in the app.
ARTIFACTS = [
    "index.json", "history.json", "macro.json", "trends.json", "monitor.json",
    "watchlist.json", "performance.json", "edge.json", "health.json", "earnings.json",
]

SHIM = """<script id="inlined-ledger-shim">
/* Standalone export. The ledger is inlined below and this shim answers the app's own
   fetches from it, so web/terminal.html runs unmodified — there is no second load path
   to drift. Anything not inlined 404s exactly as the network would. */
(function(){
  var LEDGER = __LEDGER__;
  var real = window.fetch ? window.fetch.bind(window) : null;
  function body(obj){
    return {
      ok: true, status: 200,
      json: function(){ return Promise.resolve(obj); },
      text: function(){ return Promise.resolve(JSON.stringify(obj)); }
    };
  }
  function miss(url){
    return { ok: false, status: 404,
             json: function(){ return Promise.reject(new Error('404 ' + url)); },
             text: function(){ return Promise.resolve(''); } };
  }
  window.fetch = function(input, init){
    var url = String((input && input.url) || input || '');
    var key = url.split('?')[0].replace(/^.*?ledger\\//, '').replace(/^\\.\\//, '');
    if (Object.prototype.hasOwnProperty.call(LEDGER, key)) return Promise.resolve(body(LEDGER[key]));
    if (/\\.json$/.test(url)) return Promise.resolve(miss(url));
    return real ? real(input, init) : Promise.resolve(miss(url));
  };
})();
</script>
"""


def build(out_path: str, include_symbols: int = 0) -> str:
    html = open(TERMINAL, encoding="utf-8").read()

    ledger: dict[str, object] = {}
    missing = []
    for name in ARTIFACTS:
        path = os.path.join(LEDGER, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path, encoding="utf-8") as fh:
            ledger[name] = json.load(fh)
    if "index.json" not in ledger:
        raise SystemExit("ledger/index.json is required and missing — nothing to export")

    # Optional: bake in the per-symbol files for the top N names so their tear sheets
    # render their price history offline too. Off by default; these are large.
    if include_symbols:
        top = [r["symbol"] for r in ledger["index.json"].get("top", [])][:include_symbols]
        for sym in top:
            for sub in ("history", "factors"):
                path = os.path.join(LEDGER, sub, f"{sym}.json")
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as fh:
                        ledger[f"{sub}/{sym}.json"] = json.load(fh)

    blob = json.dumps(ledger, separators=(",", ":"))
    # </script> anywhere inside the data would close the shim's own tag early.
    blob = blob.replace("</", "<\\/")
    shim = SHIM.replace("__LEDGER__", blob)

    marker = "</head>"
    if marker not in html:
        raise SystemExit("could not find </head> in the terminal — cannot install shim")
    html = html.replace(marker, shim + marker, 1)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    size = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({size / 1e6:.1f} MB)")
    print(f"  inlined: {', '.join(sorted(k for k in ledger if '/' not in k))}")
    if include_symbols:
        n = sum(1 for k in ledger if "/" in k)
        print(f"  per-symbol files: {n}")
    if missing:
        print(f"  absent from the ledger (will 404 as they would over the network): "
              f"{', '.join(missing)}")
    idx = ledger["index.json"]
    print(f"  board: {idx.get('as_of')}  model {idx.get('model_version')} "
          f"spec {idx.get('spec_hash')}  {len(idx.get('all', []))} names")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output",
                    default=os.path.join(ROOT, "Equity Conviction Monitor.html"))
    ap.add_argument("--symbols", type=int, default=0,
                    help="bake in per-symbol history/factors for the top N names")
    args = ap.parse_args(argv)
    build(args.output, args.symbols)
    return 0


if __name__ == "__main__":
    sys.exit(main())
