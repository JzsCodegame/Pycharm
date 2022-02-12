from selenium import webdriver
import time
import os
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from Chrome.ValidationMethods import ValidationMethods
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class BasicSetup:

    def passwordValid(self):

        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        driver = webdriver.Chrome(driverlocation, chrome_options=opts)
        url1 = "https://www.guru99.com/creating-keyword-hybrid-frameworks-with-selenium.html"
        url2 = ""
        driver.get(url1)
        username_1 = '//*[@id="usrname"]'
        password_1 = '//*[@id="psw"]'
        submit_1 = '/html/body/div/form/input[3]'
        time.sleep(1)
        elm_username = driver.find_element_by_xpath(username_1)
        time.sleep(1)
        elm_password = driver.find_element_by_xpath(password_1)
        elm_submit = driver.find_element_by_xpath(submit_1)
        elm_username.click()
        elm_username.send_keys('admin')
        elm_password.click()
        elm_password.send_keys('asdfg456')
        elm_submit.click()

        try:
            WebDriverWait(driver, 3).until(EC.alert_is_present(), 'Timed out waiting for PA creation ' + 'confirmation popup to appear.')

            alert = driver.switch_to.alert
            alert.accept()
            print("alert accepted")
        except TimeoutException:
            print("no alert")


run = BasicSetup()
run.passwordValid()
