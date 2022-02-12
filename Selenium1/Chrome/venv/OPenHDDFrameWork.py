from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
import os
class OPenHDDFrameWork:

    def Method2(self):
        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        browser = webdriver.Chrome(driverlocation, chrome_options=opts)
        browser.get('http://bing.com')
        js1 = 'window.open("https://www.edureka.co/blog/selenium-framework-data-keyword-hybrid-frameworks","_blank");'
        js2 = 'window.open("https://www.softwaretestingmaterial.com/data-driven-framework-selenium-webdriver/","_blank");'
        js3 = 'window.open("https://www.softwaretestinghelp.com/data-driven-framework-selenium-apache-poi/","_blank");'
        js4 = 'window.open("https://www.toolsqa.com/selenium-webdriver/data-driven-testing-excel-poi/","_blank");'
        js5 = 'window.open("https://www.toolsqa.com/selenium-webdriver/data-driven-framework/","_blank");'
        js6 = 'window.open("https://www.toolsqa.com/selenium-webdriver/data-driven-framework/","_blank");'
        browser.execute_script(js1)
        browser.execute_script(js2)
        browser.execute_script(js3)
        browser.execute_script(js4)
        browser.execute_script(js5)
        browser.execute_script(js6)

        handles = browser.window_handles
        size = len(handles)
        for x in range(size):
         browser.switch_to.window(handles[2])
        print(browser.title)

run = OPenHDDFrameWork()
run.Method2()

''' for x in range(5):
   print(runtestcase.test1()) '''