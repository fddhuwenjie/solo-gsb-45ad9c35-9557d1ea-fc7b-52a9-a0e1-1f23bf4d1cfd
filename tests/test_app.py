"""录音删段交付 API 的自动化测试。

运行：python -m pytest tests/ -v
"""

import io
import json
import math
import os
import sqlite3
import struct
import tempfile
import threading
import wave
import zipfile

# 必须在 import app 之前指向临时数据库，避免污染工作目录
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(prefix="segdel-import-"), "app.db")

import app as app_module
import pytest

RATE = 16000


def make_wav(seconds=10.0, rate=RATE, channels=1, sampwidth=2, freq=440):
    """生成测试用 WAV 字节。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        n = int(seconds * rate)
        if sampwidth == 2:
            frames = b"".join(
                struct.pack("<h", int(10000 * math.sin(2 * math.pi * freq * i / rate)))
                for i in range(n)
            )
        else:  # 8 bit 用于构造非法规格
            frames = bytes(i % 256 for i in range(n))
        if channels == 2:
            frames = b"".join(
                struct.pack("<hh", int(5000 * math.sin(2 * math.pi * freq * i / rate)), 0)
                for i in range(n)
            )
        w.writeframes(frames)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setattr(app_module, "DB_PATH", str(db_file))
    app_module.init_db()
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as c:
        yield c


def create_program(client, wav_bytes=None, chapters=None, must_keep=None):
    wav_bytes = wav_bytes if wav_bytes is not None else make_wav()
    data = {"audio": (io.BytesIO(wav_bytes), "test.wav")}
    if chapters is not None:
        data["chapters"] = json.dumps(chapters)
    if must_keep is not None:
        data["must_keep"] = json.dumps(must_keep)
    return client.post("/programs", data=data, content_type="multipart/form-data")


# ---- 创建节目 ---------------------------------------------------------------

def test_create_ok_initial_revision_keeps_everything(client):
    chapters = [{"title": "开场", "start_frame": 0}, {"title": "正文", "start_frame": 80000}]
    must_keep = [{"start_frame": 16000, "end_frame": 32000}]
    resp = create_program(client, chapters=chapters, must_keep=must_keep)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["revision"] == 0
    assert body["total_frames"] == 10 * RATE
    assert body["keep_segments"] == [{"start_frame": 0, "end_frame": 10 * RATE}]
    assert body["product_frames"] == 10 * RATE
    # 初始版本成品坐标 == 原始坐标
    assert [c["product_start_frame"] for c in body["chapters"]] == [0, 80000]
    assert body["must_keep"] == [must_keep[0]]


def test_create_rejects_bad_wav(client):
    # 非 WAV
    assert create_program(client, wav_bytes=b"not a wav").status_code == 400
    # 立体声
    resp = create_program(client, wav_bytes=make_wav(channels=2))
    assert resp.status_code == 400 and "单声道" in resp.get_json()["message"]
    # 8 kHz
    resp = create_program(client, wav_bytes=make_wav(rate=8000))
    assert resp.status_code == 400 and "16000" in resp.get_json()["message"]
    # 8 位采样
    resp = create_program(client, wav_bytes=make_wav(sampwidth=1))
    assert resp.status_code == 400 and "16" in resp.get_json()["message"]
    # 超过 30 秒
    resp = create_program(client, wav_bytes=make_wav(seconds=31))
    assert resp.status_code == 400 and "30" in resp.get_json()["message"]


def test_create_rejects_bad_chapters_and_must_keep(client):
    resp = create_program(client, chapters=[{"start_frame": 10 * RATE}])  # 越界（右开）
    assert resp.status_code == 400
    resp = create_program(client, chapters=[100, 100])  # 重复
    assert resp.status_code == 400
    resp = create_program(client, must_keep=[{"start_frame": 100, "end_frame": 100}])  # 空区间
    assert resp.status_code == 400
    resp = create_program(client, must_keep=[{"start_frame": 0, "end_frame": 10 * RATE + 1}])
    assert resp.status_code == 400


# ---- 提交保留清单 ------------------------------------------------------------

def _make_program(client):
    chapters = [
        {"title": "c0", "start_frame": 0},
        {"title": "c1", "start_frame": 40000},
        {"title": "c2", "start_frame": 48000},
        {"title": "c3", "start_frame": 80000},
        {"title": "c4", "start_frame": 88000},
        {"title": "c5", "start_frame": 120000},
    ]
    must_keep = [{"start_frame": 50000, "end_frame": 60000}]
    resp = create_program(client, chapters=chapters, must_keep=must_keep)
    assert resp.status_code == 201
    return resp.get_json()["program_id"]


def test_put_keeps_ok_and_projection(client):
    pid = _make_program(client)
    # 保留 [0,32000) [48000,88000) [112000,160000)
    resp = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 88000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["revision"] == 1
    assert body["product_frames"] == 32000 + 40000 + 48000

    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    # c0=0 在段首 -> 0；c1=40000 被删；c2=48000 位于保留段起点 -> 保留在 32000；
    # c3=80000 -> 32000+(80000-48000)=64000；c4=88000 是段2右端（右开）且不在段3 -> 删除；
    # c5=120000 -> 32000+40000+(120000-112000)=80000
    assert proj == {"c0": 0, "c2": 32000, "c3": 64000, "c5": 80000}


def test_put_keeps_stale_revision_keeps_old_state(client):
    pid = _make_program(client)
    ok = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
    })
    assert ok.status_code == 200 and ok.get_json()["revision"] == 1

    # 用过期的 0 再提交 -> 409，说明原因，状态不变
    stale = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [{"start_frame": 0, "end_frame": 160000}],
    })
    assert stale.status_code == 409
    body = stale.get_json()
    assert body["error"] == "stale_revision"
    assert body["current_revision"] == 1
    assert "0" in body["message"] and "1" in body["message"]

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert cur["keep_segments"] == [{"start_frame": 48000, "end_frame": 88000}]


def test_put_keeps_invalid_segments_rejected(client):
    pid = _make_program(client)
    cases = [
        [{"start_frame": 100, "end_frame": 100}],                      # 空段
        [{"start_frame": 200, "end_frame": 100}],                      # start > end
        [{"start_frame": -1, "end_frame": 100}],                       # 越界
        [{"start_frame": 0, "end_frame": 160001}],                     # 越界
        [{"start_frame": 50000, "end_frame": 60000},
         {"start_frame": 100, "end_frame": 200}],                      # 未升序
        [{"start_frame": 0, "end_frame": 60000},
         {"start_frame": 59000, "end_frame": 70000}],                  # 重叠
    ]
    for segs in cases:
        resp = client.put(f"/programs/{pid}/keeps",
                          json={"expected_revision": 0, "keep_segments": segs})
        assert resp.status_code == 400, segs
        assert resp.get_json()["message"]
    # 全部失败后修订号不变
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 0


def test_put_keeps_must_keep_must_be_covered(client):
    pid = _make_program(client)
    # 必留口播 [50000,60000) 被切成两段，未完整落入任何保留段
    resp = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 48000, "end_frame": 55000},
            {"start_frame": 55000, "end_frame": 88000},
        ],
    })
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "must_keep_uncovered"
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 0


def test_repeated_edits_use_original_frames_no_drift(client):
    pid = _make_program(client)
    # 第一次：删掉 [32000,48000)
    r1 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 160000},
        ],
    })
    assert r1.status_code == 200
    # 第二次：在原始坐标下进一步删掉 [96000,112000)
    r2 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 96000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    })
    assert r2.status_code == 200
    body = r2.get_json()
    assert body["revision"] == 2
    # 成品帧 = 32000 + 48000 + 48000 = 128000，与直接按原始帧计算一致
    assert body["product_frames"] == 32000 + 48000 + 48000
    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    # c5=120000 -> 32000+48000+(120000-112000)=88000
    assert proj["c5"] == 88000
    assert proj["c3"] == 32000 + (80000 - 48000)


# ---- 章节清单修改 -------------------------------------------------------------
def _put_chapters(client, pid, expected, chapters):
    return client.put(f"/programs/{pid}/chapters", json={
        "expected_revision": expected,
        "chapters": chapters,
    })


def test_put_chapters_fixes_titles_and_starts_without_audio_reupload(client):
    pid = _make_program(client)
    new_chapters = [
        {"title": "新开场", "start_frame": 0},
        {"title": "新口播", "start_frame": 50000},
        {"title": "新正文", "start_frame": 100000},
    ]
    resp = _put_chapters(client, pid, 0, new_chapters)
    assert resp.status_code == 200
    body = resp.get_json()
    # 推进一次修订；保留段原样沿用修订 0 的全段
    assert body["revision"] == 1
    assert body["keep_segments"] == [{"start_frame": 0, "end_frame": 160000}]
    assert body["product_frames"] == 160000
    # 章节替换为新清单并按保留段投影（全保留时成品坐标 == 原始坐标）
    assert body["chapters"] == [
        {"title": "新开场", "start_frame": 0, "product_start_frame": 0},
        {"title": "新口播", "start_frame": 50000, "product_start_frame": 50000},
        {"title": "新正文", "start_frame": 100000, "product_start_frame": 100000},
    ]

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert [c["title"] for c in cur["chapters"]] == ["新开场", "新口播", "新正文"]


def test_put_chapters_allows_clearing_chapters(client):
    pid = _make_program(client)
    resp = _put_chapters(client, pid, 0, [])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["revision"] == 1
    assert body["chapters"] == []
    assert body["keep_segments"] == [{"start_frame": 0, "end_frame": 160000}]
    # 清空后查询与下载都没有章节
    assert client.get(f"/programs/{pid}").get_json()["chapters"] == []
    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/programs/{pid}/download").data))
    assert json.loads(zf.read("chapters.json"))["chapters"] == []

    # 清空后可重新登记章节
    resp = _put_chapters(client, pid, 1, [{"title": "重来", "start_frame": 123}])
    assert resp.status_code == 200
    assert resp.get_json()["revision"] == 2
    proj = {c["title"]: c["product_start_frame"] for c in resp.get_json()["chapters"]}
    assert proj == {"重来": 123}


def test_put_chapters_accepts_bare_integers_and_default_titles(client):
    pid = _make_program(client)
    # 沿用创建时规则：裸整数、默认标题（按提交数组的原始序号命名）、按起点排序去重
    resp = _put_chapters(client, pid, 0, [100000, 50000])
    assert resp.status_code == 200
    assert [c["start_frame"] for c in resp.get_json()["chapters"]] == [50000, 100000]
    assert [c["title"] for c in resp.get_json()["chapters"]] == ["chapter-2", "chapter-1"]


def test_put_chapters_invalid_list_rejected_and_state_unchanged(client):
    pid = _make_program(client)
    bad_cases = [
        "not-a-list",
        [{"start_frame": 160000}],                # 越界（右开）
        [{"start_frame": -1}],                    # 越界
        [100, 100],                               # 起点重复
        [{"title": "x"}],                         # 缺起点
        ["不是整数也不是对象"],                     # 章节项类型非法
    ]
    for chapters in bad_cases:
        resp = _put_chapters(client, pid, 0, chapters)
        assert resp.status_code == 400, chapters
    # 缺字段 / 非 JSON
    resp = client.put(f"/programs/{pid}/chapters", json={"expected_revision": 0})
    assert resp.status_code == 400 and resp.get_json()["error"] == "missing_chapters"
    resp = client.put(f"/programs/{pid}/chapters",
                      data="not-json", content_type="application/json")
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_json"
    # 全部失败后章节与修订号均不变
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 0
    assert [c["title"] for c in cur["chapters"]] == ["c0", "c1", "c2", "c3", "c4", "c5"]


def test_put_chapters_stale_revision_changes_nothing(client):
    pid = _make_program(client)
    ok = _put_chapters(client, pid, 0, [{"title": "改", "start_frame": 1000}])
    assert ok.status_code == 200 and ok.get_json()["revision"] == 1

    stale = _put_chapters(client, pid, 0, [{"title": "过期", "start_frame": 2000}])
    assert stale.status_code == 409
    body = stale.get_json()
    assert body["error"] == "stale_revision"
    assert body["current_revision"] == 1

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert [c["title"] for c in cur["chapters"]] == ["改"]


def test_put_chapters_unknown_program_404(client):
    assert _put_chapters(client, "nope", 0, []).status_code == 404


def test_chapters_then_keeps_and_merge_use_commit_time_chapter_list(client):
    """章节修改之后：删段提交与旧版合入都使用提交时的章节清单。"""
    pid = _make_program(client)
    # 修订 1：章节替换为两个新章节
    assert _put_chapters(client, pid, 0, [
        {"title": "n0", "start_frame": 0},
        {"title": "n1", "start_frame": 80000},
    ]).status_code == 200

    # 修订 2：删段 [32000,48000) 与 [96000,112000)，投影使用新章节
    r2 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 96000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    })
    assert r2.status_code == 200 and r2.get_json()["revision"] == 2
    proj = {c["title"]: c["product_start_frame"] for c in r2.get_json()["chapters"]}
    # n0 在段首 0；n1=80000 落在 [48000,96000) -> 32000+32000=64000
    assert proj == {"n0": 0, "n1": 64000}
    # 旧章节标题不再出现
    assert "c3" not in proj

    # 修订 3：旧版（基于修订 1）合入删除 [128000,144000)，与修订 2 的删除不相交；
    # 合入成功响应同样使用合入提交时的章节清单（n0/n1）
    r3 = _merge(client, pid, 1, [
        {"start_frame": 0, "end_frame": 128000},
        {"start_frame": 144000, "end_frame": 160000},
    ])
    assert r3.status_code == 200 and r3.get_json()["revision"] == 3
    body3 = r3.get_json()
    proj3 = {c["title"]: c["product_start_frame"] for c in body3["chapters"]}
    assert proj3 == {"n0": 0, "n1": 64000}
    assert body3["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 128000},
        {"start_frame": 144000, "end_frame": 160000},
    ]


def test_keeps_then_chapters_reprojects_over_current_segments(client):
    """先删段、再改章节：新章节按当前保留段投影，修订号继续推进。"""
    pid = _make_program(client)
    assert client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 160000},
        ],
    }).status_code == 200

    # 新章节：一个落在被删区间 [32000,48000)（消失），一个在保留段起点（保留），
    # 一个在保留段中部
    resp = _put_chapters(client, pid, 1, [
        {"title": "a", "start_frame": 40000},
        {"title": "b", "start_frame": 48000},
        {"title": "c", "start_frame": 80000},
    ])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["revision"] == 2
    # 保留段沿用修订 1
    assert body["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 160000},
    ]
    assert body["product_frames"] == 32000 + 112000
    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    assert proj == {"b": 32000, "c": 32000 + 32000}


def test_concurrent_chapters_and_keeps_only_one_takes_effect(client):
    """并发的章节修改与删段提交基于同一 expected_revision：一个成功，另一个 409。"""
    pid = _make_program(client)
    barrier = threading.Barrier(2)
    results = {}

    def edit_chapters():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["chapters"] = _put_chapters(
                c, pid, 0, [{"title": "并发章节", "start_frame": 1000}])

    def edit_keeps():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["keeps"] = c.put(f"/programs/{pid}/keeps", json={
                "expected_revision": 0,
                "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
            })

    t1, t2 = threading.Thread(target=edit_chapters), threading.Thread(target=edit_keeps)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(r.status_code for r in results.values())
    assert statuses == [200, 409]
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    if results["chapters"].status_code == 200:
        assert [c["title"] for c in cur["chapters"]] == ["并发章节"]
        assert cur["keep_segments"] == [{"start_frame": 0, "end_frame": 160000}]
        assert results["keeps"].get_json()["error"] == "stale_revision"
    else:
        assert cur["keep_segments"] == [{"start_frame": 48000, "end_frame": 88000}]
        # 保留段 [48000,88000) 下只有 c2=48000 与 c3=80000 的章节可见
        assert [c["title"] for c in cur["chapters"]] == ["c2", "c3"]
        assert results["chapters"].get_json()["error"] == "stale_revision"


def test_concurrent_chapters_and_keeps_responses_describe_own_revision(client, monkeypatch):
    """确定性回归：先提交章节修订 1 的请求，在修订 2（删段）写入后仍返回自己的修订。

    章节修改提交修订 1 后在 COMMIT 边界阻塞，等待删段提交完整写入修订 2 并生成响应，
    再放行第一笔。修复前（章节不按修订冻结）第一笔响应会带上修订 2 的保留段与
    新章节的混合投影；修复后两笔响应分别只描述自己写入的修订。
    """
    pid = _make_program(client)
    chapters_a = [
        {"title": "a0", "start_frame": 0},
        {"title": "a1", "start_frame": 80000},
    ]
    keeps_b = [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 160000},
    ]

    a_committed = threading.Event()
    b_done = threading.Event()
    thread_role = threading.local()
    real_connect = sqlite3.connect

    class CommitHookConn:
        def __init__(self, conn):
            role = getattr(thread_role, "role", None)
            object.__setattr__(self, "_conn", conn)
            object.__setattr__(self, "_role", role)
            object.__setattr__(self, "_written_revision", None)

        def execute(self, sql, params=()):
            conn = object.__getattribute__(self, "_conn")
            cur = conn.execute(sql, params) if params else conn.execute(sql)
            text = sql.strip().upper() if isinstance(sql, str) else ""
            if object.__getattribute__(self, "_role") in ("a", "b"):
                if text.startswith("UPDATE PROGRAMS SET CURRENT_REVISION") and params:
                    object.__setattr__(self, "_written_revision", params[0])
                elif (text == "COMMIT"
                      and object.__getattribute__(self, "_role") == "a"
                      and object.__getattribute__(self, "_written_revision") == 1):
                    a_committed.set()
                    assert b_done.wait(15), "第二笔提交未在超时内完成"
            return cur

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_conn"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_conn"), name, value)

    def hooked_connect(*args, **kwargs):
        return CommitHookConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(app_module.sqlite3, "connect", hooked_connect)
    results = {}

    def edit_a():
        thread_role.role = "a"
        with app_module.app.test_client() as c:
            results["a"] = _put_chapters(c, pid, 0, chapters_a)

    def edit_b():
        thread_role.role = "b"
        assert a_committed.wait(15), "第一笔章节修改未在超时内提交"
        with app_module.app.test_client() as c:
            results["b"] = c.put(f"/programs/{pid}/keeps", json={
                "expected_revision": 1,
                "keep_segments": keeps_b,
            })
        b_done.set()

    t1 = threading.Thread(target=edit_a)
    t2 = threading.Thread(target=edit_b)
    t1.start(); t2.start()
    t1.join(20); t2.join(20)
    assert not t1.is_alive() and not t2.is_alive()
    assert results["a"].status_code == 200, results["a"].status
    assert results["b"].status_code == 200, results["b"].status

    body_a, body_b = results["a"].get_json(), results["b"].get_json()
    # 章节修订 1：全段保留，只含新章节
    assert body_a["revision"] == 1
    assert body_a["keep_segments"] == _revision_snapshot(pid, 1)["keep_segments"]
    assert body_a["chapters"] == _revision_snapshot(pid, 1)["chapters"]
    assert body_a["product_frames"] == 160000
    assert {c["title"] for c in body_a["chapters"]} == {"a0", "a1"}
    # 删段修订 2：沿用提交时的新章节，按新保留段投影
    assert body_b["revision"] == 2
    assert body_b["keep_segments"] == keeps_b
    assert body_b["chapters"] == _revision_snapshot(pid, 2)["chapters"]
    proj_b = {c["title"]: c["product_start_frame"] for c in body_b["chapters"]}
    assert proj_b == {"a0": 0, "a1": 32000 + (80000 - 48000)}

    # 当前为修订 2；修订 1 的快照仍可复现（历史修订不可变）
    assert _revision_snapshot(pid, 1)["product_frames"] == 160000
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2 and cur["keep_segments"] == keeps_b


def test_legacy_db_without_revision_chapters_is_migrated_and_editable(monkeypatch, tmp_path):
    """升级前建的库（无 revision_chapters）：回填历史快照后节目可继续编辑。"""
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy))
    conn.executescript("""
    CREATE TABLE programs (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, sample_rate INTEGER NOT NULL,
        channels INTEGER NOT NULL, sample_width INTEGER NOT NULL, total_frames INTEGER NOT NULL,
        wav BLOB NOT NULL, current_revision INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE chapters (
        program_id TEXT NOT NULL, idx INTEGER NOT NULL, title TEXT NOT NULL,
        start_frame INTEGER NOT NULL, PRIMARY KEY (program_id, idx));
    CREATE TABLE must_keeps (
        program_id TEXT NOT NULL, idx INTEGER NOT NULL, start_frame INTEGER NOT NULL,
        end_frame INTEGER NOT NULL, PRIMARY KEY (program_id, idx));
    CREATE TABLE revisions (
        program_id TEXT NOT NULL, revision INTEGER NOT NULL, created_at TEXT NOT NULL,
        PRIMARY KEY (program_id, revision));
    CREATE TABLE revision_segments (
        program_id TEXT NOT NULL, revision INTEGER NOT NULL, idx INTEGER NOT NULL,
        start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
        PRIMARY KEY (program_id, revision, idx));
    """)
    conn.execute(
        "INSERT INTO programs VALUES ('pid1','t',16000,1,2,1000,x'',1)")
    conn.executemany(
        "INSERT INTO chapters (program_id, idx, title, start_frame) VALUES (?,?,?,?)",
        [("pid1", 0, "旧章节", 100), ("pid1", 1, "旧章节2", 600)])
    conn.execute("INSERT INTO revisions VALUES ('pid1',0,'t')")
    conn.execute("INSERT INTO revision_segments VALUES ('pid1',0,0,0,1000)")
    conn.execute("INSERT INTO revisions VALUES ('pid1',1,'t')")
    conn.executemany(
        "INSERT INTO revision_segments VALUES (?,?,?,?,?)",
        [("pid1", 1, 0, 0, 400), ("pid1", 1, 1, 500, 1000)])
    conn.commit()
    conn.close()

    monkeypatch.setattr(app_module, "DB_PATH", str(legacy))
    app_module.init_db()
    # 再执行一次：回填幂等，不产生重复行
    app_module.init_db()
    app_module.app.config.update(TESTING=True)

    with app_module.app.test_client() as c:
        cur = c.get("/programs/pid1").get_json()
        assert cur["revision"] == 1
        # 修订 1 的历史投影与升级前一致：600 落在被删区间 [400,500) 之前的段 [0,400)?
        # 600 落在 [500,1000) -> 成品 400+(600-500)=500；100 -> 100
        proj = {ch["title"]: ch["product_start_frame"] for ch in cur["chapters"]}
        assert proj == {"旧章节": 100, "旧章节2": 500}

        # 已有节目可继续编辑：改章节
        resp = c.put("/programs/pid1/chapters", json={
            "expected_revision": 1,
            "chapters": [{"title": "迁移后", "start_frame": 550}],
        })
        assert resp.status_code == 200 and resp.get_json()["revision"] == 2
        proj = {ch["title"]: ch["product_start_frame"]
                for ch in resp.get_json()["chapters"]}
        assert proj == {"迁移后": 400 + 50}

        # 也可继续删段
        resp = c.put("/programs/pid1/keeps", json={
            "expected_revision": 2,
            "keep_segments": [{"start_frame": 0, "end_frame": 400}],
        })
        assert resp.status_code == 200 and resp.get_json()["revision"] == 3
        assert resp.get_json()["chapters"] == []  # 550 不在 [0,400)


def test_download_after_chapter_edit_uses_same_revision(client):
    """章节修改后下载：WAV 与 chapters.json 仍取自同一修订快照。"""
    pid = _make_program(client)
    # 先删段（修订 1）
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 160000},
        ],
    })
    # 再改章节（修订 2）：标题/起点全部替换
    assert _put_chapters(client, pid, 1, [
        {"title": "下载章节", "start_frame": 80000},
    ]).status_code == 200

    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/programs/{pid}/download").data))
    doc = json.loads(zf.read("chapters.json"))
    assert doc["revision"] == 2
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        wav_frames = w.getnframes()
    assert wav_frames == doc["product_frames"] == 32000 + 112000
    assert doc["chapters"] == [{"title": "下载章节", "start_frame": 64000}]

# ---- 下载 --------------------------------------------------------------------

def test_download_zip_consistent_with_revision(client):
    pid = _make_program(client)
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 88000},
        ],
    })
    resp = client.get(f"/programs/{pid}/download")
    assert resp.status_code == 200
    assert resp.mimetype == "application/zip"
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    assert set(zf.namelist()) == {"audio.wav", "chapters.json"}

    # WAV 是真正裁接过的：帧数 == 保留段总长
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        assert w.getframerate() == RATE
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 32000 + 40000

    doc = json.loads(zf.read("chapters.json"))
    assert doc["revision"] == 1
    assert doc["product_frames"] == 32000 + 40000
    starts = {c["title"]: c["start_frame"] for c in doc["chapters"]}
    assert starts == {"c0": 0, "c2": 32000, "c3": 64000}

    # 再次编辑后下载：WAV 与 JSON 同属修订 2
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
    })
    resp = client.get(f"/programs/{pid}/download")
    zf = zipfile.ZipFile(io.BytesIO(resp.data))
    doc = json.loads(zf.read("chapters.json"))
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        wav_frames = w.getnframes()
    assert doc["revision"] == 2
    assert wav_frames == doc["product_frames"] == 40000
    starts = {c["title"]: c["start_frame"] for c in doc["chapters"]}
    assert starts == {"c2": 0, "c3": 32000}


def test_download_wav_content_matches_original_splice(client):
    """裁接结果应与原始音频对应帧完全一致（原始音频不变）。"""
    wav_bytes = make_wav(seconds=2)
    resp = create_program(client, wav_bytes=wav_bytes)
    pid = resp.get_json()["program_id"]
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 100, "end_frame": 200},
            {"start_frame": 300, "end_frame": 500},
        ],
    })
    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/programs/{pid}/download").data))
    product = zf.read("audio.wav")
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        w.setpos(100)
        expect = w.readframes(100)
        w.setpos(300)
        expect += w.readframes(200)
    with wave.open(io.BytesIO(product), "rb") as w:
        actual = w.readframes(w.getnframes())
    assert actual == expect


# ---- 持久化 ------------------------------------------------------------------

def test_program_survives_reconnect(client, tmp_path, monkeypatch):
    """模拟重启：所有状态都在 SQLite 文件中，新连接可继续编辑。"""
    pid = _make_program(client)
    client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [{"start_frame": 48000, "end_frame": 88000}],
    })
    # 重新指向同一个数据库文件并重新初始化（等价于进程重启后打开已有库）
    db_path = app_module.DB_PATH
    app_module.init_db()
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    nxt = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [{"start_frame": 50000, "end_frame": 60000}],
    })
    assert nxt.status_code == 200 and nxt.get_json()["revision"] == 2


def test_unknown_program_404(client):
    assert client.get("/programs/nope").status_code == 404
    assert client.put("/programs/nope/keeps",
                      json={"expected_revision": 0, "keep_segments": []}).status_code == 404
    assert client.get("/programs/nope/download").status_code == 404


# ---- 旧版建议合入（三方合入）---------------------------------------------------
def _merge(client, pid, base, segments):
    return client.post(f"/programs/{pid}/merges", json={
        "base_revision": base,
        "keep_segments": segments,
    })


def _setup_two_branches(client):
    """基版修订 0 全保留；修订 1 由当前用户删去 [32000,48000)。

    旧版建议仍以修订 0 为依据，可删与之不相交的其它区间。
    """
    pid = _make_program(client)
    r1 = client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 0,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 160000},
        ],
    })
    assert r1.status_code == 200 and r1.get_json()["revision"] == 1
    return pid


def test_merge_disjoint_changes_synthesizes_revision(client):
    pid = _setup_two_branches(client)
    # 建议（依据修订 0）删去 [96000,112000)，与当前已删的 [32000,48000) 互不相交
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["revision"] == 2
    assert body["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert body["product_frames"] == 32000 + 48000 + 48000
    # 章节查询反映合并后的同一修订：c5=120000 落在 [112000,160000)
    proj = {c["title"]: c["product_start_frame"] for c in body["chapters"]}
    assert proj["c5"] == 32000 + 48000 + (120000 - 112000)
    assert "c1" not in proj  # 40000 落在当前侧删除区


def test_merge_seen_after_get_and_download(client):
    pid = _setup_two_branches(client)
    merged_segments = [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]).status_code == 200

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2 and cur["keep_segments"] == merged_segments

    zf = zipfile.ZipFile(io.BytesIO(client.get(f"/programs/{pid}/download").data))
    with wave.open(io.BytesIO(zf.read("audio.wav")), "rb") as w:
        wav_frames = w.getnframes()
    doc = json.loads(zf.read("chapters.json"))
    assert doc["revision"] == 2
    assert wav_frames == doc["product_frames"] == 32000 + 48000 + 48000
    starts = {c["title"]: c["start_frame"] for c in doc["chapters"]}
    assert starts["c5"] == 32000 + 48000 + 8000


def test_merge_overlapping_changes_conflict_409(client):
    pid = _setup_two_branches(client)
    # 建议删 [40000,56000)，与当前删除 [32000,48000) 相交于 [40000,48000)
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 40000},
        {"start_frame": 56000, "end_frame": 160000},
    ])
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error"] == "merge_conflict"
    assert body["base_revision"] == 0
    assert body["current_revision"] == 1
    assert body["conflicts"] == [{"start_frame": 40000, "end_frame": 48000}]
    # 当前版本不变
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert cur["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 160000},
    ]


def test_merge_both_sides_delete_same_segment_is_conflict(client):
    pid = _setup_two_branches(client)
    # 建议清单与当前修订完全相同：双方相对基版都删了 [32000,48000)，不能擅自视为一致
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 160000},
    ])
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error"] == "merge_conflict"
    assert body["conflicts"] == [{"start_frame": 32000, "end_frame": 48000}]
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_adjacent_but_disjoint_changes_ok(client):
    pid = _setup_two_branches(client)
    # 当前删 [32000,48000)；建议删 [16000,32000)——仅相邻、帧区间不相交
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 160000},
    ])
    assert resp.status_code == 200
    assert resp.get_json()["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 48000, "end_frame": 160000},
    ]


def test_merge_base_revision_missing_404(client):
    pid = _setup_two_branches(client)
    resp = _merge(client, pid, 7, [
        {"start_frame": 0, "end_frame": 160000},
    ])
    assert resp.status_code == 404
    assert resp.get_json()["error"] == "base_revision_not_found"
    assert resp.get_json()["current_revision"] == 1
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_bad_base_revision_field_400(client):
    pid = _setup_two_branches(client)
    resp = client.post(f"/programs/{pid}/merges", json={
        "base_revision": "0",
        "keep_segments": [{"start_frame": 0, "end_frame": 160000}],
    })
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_base_revision"

    resp = client.post(f"/programs/{pid}/merges", json={
        "keep_segments": [{"start_frame": 0, "end_frame": 160000}],
    })
    assert resp.status_code == 400 and resp.get_json()["error"] == "missing_base_revision"
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_invalid_suggested_segments_400(client):
    pid = _setup_two_branches(client)
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 60000},
        {"start_frame": 59000, "end_frame": 70000},  # 重叠、非升序
    ])
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_segments"

    resp = _merge(client, pid, 0, [{"start_frame": -1, "end_frame": 100}])
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_segments"

    resp = client.post(f"/programs/{pid}/merges",
                       data="not-json", content_type="application/json")
    assert resp.status_code == 400 and resp.get_json()["error"] == "invalid_json"
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1


def test_merge_result_must_cover_must_keep_400(client):
    pid = _setup_two_branches(client)
    # 必留口播 [50000,60000)；建议删 [48000,96000)，与当前删除不相交，
    # 但合并结果把口播整段删掉 -> 拒绝，当前版本不变
    resp = _merge(client, pid, 0, [
        {"start_frame": 0, "end_frame": 48000},
        {"start_frame": 96000, "end_frame": 160000},
    ])
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "must_keep_uncovered"
    assert body["must_keep"] == {"start_frame": 50000, "end_frame": 60000}
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1


def test_merge_against_nonzero_base_with_independent_followups(client):
    pid = _setup_two_branches(client)
    # 再以修订 1 为基版：当前侧（修订 2）删 [96000,112000)，建议侧删 [120000,130000)
    assert client.put(f"/programs/{pid}/keeps", json={
        "expected_revision": 1,
        "keep_segments": [
            {"start_frame": 0, "end_frame": 32000},
            {"start_frame": 48000, "end_frame": 96000},
            {"start_frame": 112000, "end_frame": 160000},
        ],
    }).status_code == 200
    resp = _merge(client, pid, 1, [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 120000},
        {"start_frame": 130000, "end_frame": 160000},
    ])
    assert resp.status_code == 200
    assert resp.get_json()["keep_segments"] == [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 120000},
        {"start_frame": 130000, "end_frame": 160000},
    ]


def test_merge_unknown_program_404(client):
    assert client.post("/programs/nope/merges", json={
        "base_revision": 0,
        "keep_segments": [{"start_frame": 0, "end_frame": 100}],
    }).status_code == 404


def test_concurrent_merges_both_take_effect(client):
    """两个并发合入都基于修订 0、删除互不相交的区间：串行化后两处删除都生效。"""
    pid = _make_program(client)  # 必留口播 [50000,60000)
    barrier = threading.Barrier(2)
    results = {}

    def merge_a():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["a"] = _merge(c, pid, 0, [
                {"start_frame": 0, "end_frame": 16000},
                {"start_frame": 32000, "end_frame": 160000},
            ])

    def merge_b():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["b"] = _merge(c, pid, 0, [
                {"start_frame": 0, "end_frame": 96000},
                {"start_frame": 112000, "end_frame": 160000},
            ])

    t1, t2 = threading.Thread(target=merge_a), threading.Thread(target=merge_b)
    t1.start(); t2.start(); t1.join(); t2.join()

    assert results["a"].status_code == 200, results["a"].status
    assert results["b"].status_code == 200, results["b"].status
    assert {results["a"].get_json()["revision"],
            results["b"].get_json()["revision"]} == {1, 2}

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2
    # 两处删除 [16000,32000) 与 [96000,112000) 都在最终修订中
    assert cur["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert cur["product_frames"] == 16000 + 64000 + 48000


def _revision_snapshot(pid, revision):
    """直接从不可变修订表读出该修订的保留段、成品帧数与章节投影，作为期望。"""
    conn = sqlite3.connect(app_module.DB_PATH)
    try:
        seg_rows = conn.execute(
            "SELECT start_frame, end_frame FROM revision_segments "
            "WHERE program_id = ? AND revision = ? ORDER BY idx",
            (pid, revision),
        ).fetchall()
        segments = [(s, e) for s, e in seg_rows]
        chapter_rows = conn.execute(
            "SELECT title, start_frame FROM revision_chapters "
            "WHERE program_id = ? AND revision = ? ORDER BY idx",
            (pid, revision),
        ).fetchall()
        chapters = [{"title": t, "start_frame": f} for t, f in chapter_rows]
    finally:
        conn.close()
    return {
        "revision": revision,
        "keep_segments": [{"start_frame": s, "end_frame": e} for s, e in segments],
        "product_frames": sum(e - s for s, e in segments),
        "chapters": app_module.project_chapters(chapters, segments),
    }


def test_concurrent_merges_response_describes_own_revision(client, monkeypatch):
    """确定性回归：先提交修订 1 的合入，必须在修订 2 已写入后仍返回自己的修订。

    在数据库连接层面拦截合入事务的 COMMIT（该边界在修复前后代码中完全一致）：
    提交修订 1 的请求在 COMMIT 之后、生成响应之前阻塞，确定地等待第二笔合入
    完整完成（拿到写锁、合入、提交修订 2 并生成响应）后才继续。
    修复前第一笔会重新读取此刻的当前修订 2，于是两笔响应都返回修订 2 的
    保留段/成品帧数/章节投影；修复后第一笔仍返回自己写入的修订 1。
    """
    pid = _make_program(client)  # 必留口播 [50000,60000)，章节见 _make_program
    segs_a = [  # 删除 [16000,32000)
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 160000},
    ]
    segs_b = [  # 删除 [96000,112000)，与 A 不相交
        {"start_frame": 0, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]

    a_committed = threading.Event()  # 第一笔（A）已提交修订 1
    b_done = threading.Event()       # 第二笔（B）已提交修订 2 且响应已完整生成
    thread_role = threading.local()
    real_connect = sqlite3.connect

    class CommitHookConn:
        """连接代理：只在合入线程的提交点注入阻塞，其余行为原样透传。"""

        def __init__(self, conn):
            role = getattr(thread_role, "role", None)
            object.__setattr__(self, "_conn", conn)
            object.__setattr__(self, "_role", role)
            object.__setattr__(self, "_written_revision", None)

        def execute(self, sql, params=()):
            conn = object.__getattribute__(self, "_conn")
            cur = conn.execute(sql, params) if params else conn.execute(sql)
            text = sql.strip().upper() if isinstance(sql, str) else ""
            if object.__getattribute__(self, "_role") in ("a", "b"):
                if text.startswith("UPDATE PROGRAMS SET CURRENT_REVISION") and params:
                    object.__setattr__(self, "_written_revision", params[0])
                elif (text == "COMMIT"
                      and object.__getattribute__(self, "_role") == "a"
                      and object.__getattribute__(self, "_written_revision") == 1):
                    # A 已提交修订 1、响应尚未生成：放行 B，确定地等 B 完整完成后再继续。
                    a_committed.set()
                    assert b_done.wait(15), "第二笔合入未在超时内完成"
            return cur

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_conn"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_conn"), name, value)

    def hooked_connect(*args, **kwargs):
        return CommitHookConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(app_module.sqlite3, "connect", hooked_connect)

    results = {}

    def merge_a():
        thread_role.role = "a"
        with app_module.app.test_client() as c:
            results["a"] = _merge(c, pid, 0, segs_a)

    def merge_b():
        thread_role.role = "b"
        # 在 A 提交修订 1 之前绝不发起请求，保证 A 拿到第一个写锁成为修订 1
        assert a_committed.wait(15), "第一笔合入未在超时内提交"
        with app_module.app.test_client() as c:
            results["b"] = _merge(c, pid, 0, segs_b)
        # 第二笔响应已完整生成后才放行第一笔
        b_done.set()

    t1 = threading.Thread(target=merge_a)
    t2 = threading.Thread(target=merge_b)
    t1.start(); t2.start()
    t1.join(20); t2.join(20)
    assert not t1.is_alive() and not t2.is_alive()

    assert results["a"].status_code == 200, results["a"].status
    assert results["b"].status_code == 200, results["b"].status

    # 两笔响应必须各自携带不同的修订号（修复前第一笔会错误地也返回修订 2）
    resp_by_revision = {r.get_json()["revision"]: r.get_json()
                        for r in (results["a"], results["b"])}
    assert sorted(resp_by_revision) == [1, 2]

    # 每笔成功响应都对应自己写入的修订：修订号、保留段、成品帧数、章节投影一致
    for rev, body in sorted(resp_by_revision.items()):
        expected = _revision_snapshot(pid, rev)
        assert body["revision"] == rev
        assert body["keep_segments"] == expected["keep_segments"]
        assert body["product_frames"] == expected["product_frames"]
        assert body["chapters"] == expected["chapters"]

    # 修订 1 只含第一笔自己的删除，修订 2 含两笔非冲突修改
    first, second = resp_by_revision[1], resp_by_revision[2]
    assert first["keep_segments"] == segs_a
    assert first["product_frames"] == 16000 + 128000
    assert second["keep_segments"] == [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 96000},
        {"start_frame": 112000, "end_frame": 160000},
    ]
    assert second["product_frames"] == 16000 + 64000 + 48000

    # 最终当前版本仍包含两笔非冲突修改，章节投影按最终修订
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 2
    assert cur["keep_segments"] == second["keep_segments"]
    assert cur["product_frames"] == second["product_frames"]
    assert cur["chapters"] == _revision_snapshot(pid, 2)["chapters"]


def test_concurrent_merges_same_deletion_one_conflict_no_write(client):
    """两笔基于同一旧修订、删除同一区间的并发合入：一个成功，另一个 409 且不写入。"""
    pid = _make_program(client)
    same = [
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 160000},
    ]
    barrier = threading.Barrier(2)
    results = {}

    def merge_a():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["a"] = _merge(c, pid, 0, same)

    def merge_b():
        with app_module.app.test_client() as c:
            barrier.wait()
            results["b"] = _merge(c, pid, 0, same)

    t1, t2 = threading.Thread(target=merge_a), threading.Thread(target=merge_b)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(r.status_code for r in results.values())
    assert statuses == [200, 409]
    conflict = next(r for r in results.values() if r.status_code == 409).get_json()
    assert conflict["error"] == "merge_conflict"
    assert conflict["current_revision"] == 1
    assert conflict["conflicts"] == [{"start_frame": 16000, "end_frame": 32000}]

    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert cur["keep_segments"] == same


def test_concurrent_merges_must_keep_failure_rejected_no_write(client):
    """并发合入：删段会破坏必留口播的一笔即使在另一笔成功后执行，仍 400 且不写入。"""
    pid = _make_program(client)  # 必留口播 [50000,60000)
    good = [  # 删 [16000,32000)，口播仍完整保留
        {"start_frame": 0, "end_frame": 16000},
        {"start_frame": 32000, "end_frame": 160000},
    ]
    bad = [  # 删 [48000,96000)，口播 [50000,60000) 被整体删除
        {"start_frame": 0, "end_frame": 48000},
        {"start_frame": 96000, "end_frame": 160000},
    ]
    # 先让一笔合规合入生效，再以修订 0 为基版提交坏建议：
    # 坏建议相对基版的改动与修订 1 的删除不相交，因此唯一拒绝原因是必留口播。
    ok = _merge(client, pid, 0, good)
    assert ok.status_code == 200 and ok.get_json()["revision"] == 1
    resp = _merge(client, pid, 0, bad)
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "must_keep_uncovered"
    assert body["must_keep"] == {"start_frame": 50000, "end_frame": 60000}
    cur = client.get(f"/programs/{pid}").get_json()
    assert cur["revision"] == 1
    assert cur["keep_segments"] == good


def test_concurrent_puts_winner_response_describes_own_revision(client):
    """并发提交同一 expected_revision：赢家返回自己写入的修订，输家 409 不写入。"""
    pid = _make_program(client)
    barrier = threading.Barrier(2)
    results = {}
    segs = [
        {"start_frame": 0, "end_frame": 32000},
        {"start_frame": 48000, "end_frame": 88000},
        {"start_frame": 112000, "end_frame": 160000},
    ]

    def put(tag):
        with app_module.app.test_client() as c:
            barrier.wait()
            results[tag] = c.put(f"/programs/{pid}/keeps", json={
                "expected_revision": 0,
                "keep_segments": segs,
            })

    t1, t2 = threading.Thread(target=put, args=("a",)), threading.Thread(target=put, args=("b",))
    t1.start(); t2.start(); t1.join(); t2.join()

    winner = next(r for r in results.values() if r.status_code == 200).get_json()
    loser = next(r for r in results.values() if r.status_code == 409).get_json()
    assert winner["revision"] == 1
    assert winner["keep_segments"] == segs
    assert winner["product_frames"] == 32000 + 40000 + 48000
    assert winner["chapters"] == _revision_snapshot(pid, 1)["chapters"]
    assert loser["error"] == "stale_revision" and loser["current_revision"] == 1
    assert client.get(f"/programs/{pid}").get_json()["revision"] == 1
