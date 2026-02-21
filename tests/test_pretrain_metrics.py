from src.pipeline.pretrain import evaluate_pretrain_metrics


def test_pretrain_metric_eval_pass():
    summary = {
        "avg_mine_per_episode": 3.2,
        "avg_steps": 7.1,
        "avg_buy_water": 140.0,
        "avg_buy_food": 120.0,
        "success_rate": 1.0,
        "avg_value_loss": 120.0,
        "final_value_loss": 80.0,
        "value_loss_trend": -50.0,
        "match_rate": 0.55,
    }

    report = evaluate_pretrain_metrics(summary)

    assert report["overall_pass"] is True
    assert report["redline"] is False
    assert all(item["status"] == "PASS" for item in report["checks"])


def test_pretrain_metric_eval_redline_detected():
    summary = {
        "avg_mine_per_episode": 0.0,
        "avg_steps": 4.0,
        "avg_buy_water": 80.0,
        "avg_buy_food": 80.0,
        "success_rate": 1.0,
        "avg_value_loss": 650.0,
        "final_value_loss": 700.0,
        "value_loss_trend": 120.0,
        "match_rate": 0.15,
    }

    report = evaluate_pretrain_metrics(summary)

    assert report["overall_pass"] is False
    assert report["redline"] is True
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["矿山访问率"] == "FAIL"
    assert statuses["步长分布"] == "FAIL"
    assert statuses["资源购买量"] == "FAIL"
    assert statuses["成功率与价值收敛"] == "FAIL"
