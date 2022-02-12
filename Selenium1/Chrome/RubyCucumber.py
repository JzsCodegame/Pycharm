from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
import os
class RubyCucumber:
    def Method4(self):
        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        browser = webdriver.Chrome(driverlocation, chrome_options=opts)
        browser.get('http://bing.com')
        js1 = 'window.open("https://www.agiratech.com/blog/web-automation-testing-with-ruby-cucumber-watir","_blank");'
        js2 = 'window.open("https://codoid.com/creating-a-ruby-cucumber-project/","_blank");'
        js3 = 'window.open("https://www.guru99.com/your-first-cucumber-script.html","_blank");'
        js4 = 'window.open("https://www.jetbrains.com/help/ruby/get-started.html#open","_blank");'
        js5 = 'window.open("https://www.activestate.com/products/komodo-ide/download-ide/","_blank");'
        js6 = 'window.open("https://www.tutorialspoint.com/cucumber/cucumber_gherkins.htm","_blank");'
        js7 = 'window.open("https://www.guru99.com/introduction-to-cucumber.html","_blank");'
        js8 = 'window.open("https://www.guru99.com/gherkin-test-cucumber.html","_blank");'
        browser.execute_script(js1)
        browser.execute_script(js2)
        browser.execute_script(js3)
        browser.execute_script(js4)


run = RubyCucumber()
run.Method4()

''' for x in range(5):
   print(runtestcase.test1()) '''