from selenium import webdriver
import os
from selenium.webdriver import ChromeOptions

class BasicSetup:

    def passwordValid(self):

        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        driver = webdriver.Chrome(driverlocation, chrome_options=opts)
        url1 = "http://automationpractice.com/index.php"
        driver.get(url1)
        elm_signin = driver.find_element_by_xpath("//*[@class='login']")
        elm_signin.click()
        email_address = driver.find_element_by_xpath("//*[@name='email_create']")
        email_address.send_keys('admin')
        create_account = driver.find_element_by_xpath("//*[@name='SubmitCreate']")
        create_account.send_keys('asdfg456')
        create_account.click()



runtestcase1 = BasicSetup()

x = runtestcase1.passwordValid()

for x in range(5):
    print(runtestcase1.passwordValid())

