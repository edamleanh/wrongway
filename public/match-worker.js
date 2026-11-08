// Chạy một trận bot vs bot trong Web Worker bằng Pyodide (Python trên WebAssembly).
// Code bot chạy trong sandbox của trình duyệt, không chạy trên server.
// Luật lấy từ chính environment.py của Arena; xử lý lượt giống Arena:
// lỗi / quá giờ / hành động sai -> mất lượt, ván vẫn tiếp tục.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

const HARNESS = `
import importlib.util, json, random, sys, time, traceback
sys.path.insert(0, "/arena")
from environment import Environment

bots = [None, None]
env = None

def load_bot(i, path):
    spec = importlib.util.spec_from_file_location("bot_slot_%d" % i, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "choose_action", None)):
        raise ValueError("Thiếu hàm choose_action(observation)")
    bots[i] = module

def start(max_turns, seed):
    global env
    random.seed(seed)
    env = Environment(max_turns=max_turns)
    return json.dumps(env.snapshot())

def step(limit):
    p = env.current_player
    obs = env.get_observation(p)
    error = None
    t0 = time.perf_counter()
    try:
        action = bots[p].choose_action(obs)
    except Exception:
        action = None
        error = traceback.format_exc(limit=4)
    elapsed = time.perf_counter() - t0
    if error is not None:
        result = env.skip_turn("error"); status = "error"
    elif elapsed > limit:
        result = env.skip_turn("timeout"); status = "timeout"
    else:
        try:
            action = json.loads(json.dumps(action))   # chỉ chấp nhận dữ liệu kiểu JSON
        except Exception:
            action = "not-json"
        result = env.step(action)
        status = result["status"]
    return json.dumps({"player": p, "status": status, "action": result.get("action"),
                       "reason": result.get("reason"), "elapsed": round(elapsed, 3),
                       "error": error, "state": env.snapshot()})
`;

let pyodide = null;
const boot = (async () => {
  pyodide = await loadPyodide();
  pyodide.setStdout({ batched: () => {} });   // bỏ qua print() của bot
  pyodide.setStderr({ batched: () => {} });
})();

onmessage = async (event) => {
  const msg = event.data;
  try {
    await boot;
    if (msg.type === "init") {
      pyodide.FS.mkdirTree("/arena");
      pyodide.FS.mkdirTree("/bots");
      pyodide.FS.writeFile("/arena/environment.py", msg.env);
      pyodide.runPython(HARNESS);
      for (const i of [0, 1]) {
        pyodide.FS.writeFile("/bots/bot" + i + ".py", msg.codes[i]);
        try {
          pyodide.globals.get("load_bot")(i, "/bots/bot" + i + ".py");
        } catch (err) {
          postMessage({ type: "load-error", slot: i, message: String(err.message || err) });
          return;
        }
      }
      const state = JSON.parse(pyodide.globals.get("start")(msg.maxTurns, msg.seed));
      postMessage({ type: "ready", state });
    } else if (msg.type === "step") {
      postMessage({ type: "step", record: JSON.parse(pyodide.globals.get("step")(msg.limit)) });
    }
  } catch (err) {
    postMessage({ type: "error", message: String(err.message || err) });
  }
};
