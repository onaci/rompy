import os
import shutil
from datetime import datetime

import pytest
from utils import compare_files

from rompy.core import BaseConfig, BaseModel, TimeRange

here = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def model():
    return BaseModel(
        run_id="test_base",
        output_dir=os.path.join(here, "simulations"),
        # template=BaseConfig(),
    )


@pytest.fixture
def gitlab_template():
    return BaseModel(
        template=BaseConfig(
            template="git@gitlab.com:oceanum/models/test-rompy-template.git",
        )
    )


def test_datetime_parse():
    end = datetime(2022, 2, 21, 4)
    for format in [
        "%Y%m%d.%H%M%S",
        "%Y%m%d.%H%M",
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H%M",
    ]:
        model = BaseModel(period=TimeRange(end=end.strftime(format), duration="1d"))
        for period in ["year", "month", "day", "hour"]:
            assert getattr(model.period.end, period) == getattr(end, period)


def test_datetime_parse_fail():
    end = datetime(2022, 2, 21, 4)
    for format in [
        "%Y%m%d.%Hhello",
        "%Y%m%dhello",
    ]:
        try:
            model = BaseModel(period=TimeRange(end=end.strftime(format), duration="1d"))
        except ValueError:
            pass
        else:
            raise ValueError("Should not be able to parse {format}")


# test generate
def test_generate(model):
    model.generate()
    compare_files(
        os.path.join(here, "template_output/test_base/INPUT"),
        os.path.join(here, "simulations/test_base_ref/INPUT"),
    )


# repeat suite for gitlab template
@pytest.mark.skip(reason="gitlab template not ready following restructure")
def test_gitlab_template(gitlab_template):
    gitlab_template.generate()
    compare_files(
        os.path.join(here, "template_output/test_base/INPUT"),
        os.path.join(here, "simulations/test_base_ref/INPUT"),
    )
