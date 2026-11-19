"""Smoke tests: every page renders without exceptions in each language."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = ["views/home.py", "views/predict.py", "views/market.py", "views/search.py", "views/sell.py",
         "views/invest.py", "views/insights.py", "views/account.py"]


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HOUSEFINDER_DB", str(tmp_path / "app.db"))
    from housefinder import config
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "app.db")


@pytest.mark.parametrize("lang", ["uz", "en", "ru"])
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, lang):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["lang"] = lang
    at.run()
    at.switch_page(page).run()
    assert not at.exception, at.exception


def test_register_login_and_publish():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("views/account.py").run()
    at.text_input(key="r_user").input("tester")
    at.text_input(key="r_pass").input("secret12")
    at.text_input(key="r_pass2").input("secret12")
    at.button[-1].click()  # register is the last form button
    at.run()
    assert not at.exception
    assert at.session_state["user"]["username"] == "tester"

    at.switch_page("views/sell.py").run()
    at.text_input(key="sl_title").input("Ajoyib uy, bog‘ bilan")
    at.number_input(key="sl_price").set_value(185000)
    publish = [b for b in at.button if "📢" in b.label][0]
    publish.click()
    at.run()
    assert not at.exception
    assert any("#" in s.value for s in at.success)

    at.switch_page("views/account.py").run()
    assert not at.exception
    at.switch_page("views/search.py").run()
    assert not at.exception
