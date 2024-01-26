#!/usr/bin/env python

# Copyright 2022 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import final

from datetime import timedelta, date
from locust import FastHttpUser, TaskSet, task, between
import pandas as pd
import random


# [START locust_test_task]

class MetricsTaskSet(TaskSet):
    _countries = ['Australia', 'Brazil', 'Canada', 'France', 'Germany', 'India', 'Italy', 'Netherlands', 'Poland', 'Russian Federation', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom of Great Britain and Northern Ireland', 'United States of America']
    _educations = ['Bachelor’s degree', 'Less than a Bachelors', 'Master’s degree', 'Post grad']
    _genders = ['Man', 'Woman', 'Other']

    wait_time = between(1, 5)


    def on_start(self):
        # Initialize some random data
        self._country = random.choice(self._countries)
        self._age = random.randint(18, 65)
        self._gender = random.choice(self._genders)
        self._education_level = random.choice(self._educations)
        self._years_of_experience = random.randint(0, self._age - 1)
        

    @task
    def predict_salary(self):
        myheaders = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        self.client.post(
            '/prediction',
            json={
                "country": self._country,
                "age": self._age,
                "gender": self._gender,
                "education": self._education_level,
                "experience": self._years_of_experience
            },
            headers=myheaders
        )

class MetricsLocust(FastHttpUser):
    tasks = {MetricsTaskSet}

# [END locust_test_task]
