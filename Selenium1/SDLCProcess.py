from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
import os
class SDLCProcess:
    def Method3(self):
        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        browser = webdriver.Chrome(driverlocation, chrome_options=opts)
        browser.get('http://bing.com')
        js1 = 'window.open("https://sites.google.com/site/ustcomp2com/sdlc-stages","_blank");'
        js2 = 'window.open("https://www.softwaretestinghelp.com/software-development-life-cycle-sdlc/","_blank");'
        js3 = 'window.open("https://www.tutorialspoint.com/system_analysis_and_design/system_analysis_and_design_development_life_cycle.htm","_blank");'
        js4 = 'window.open("https://www.toolsqa.com/selenium-webdriver/data-driven-testing-excel-poi/","_blank");'
        browser.execute_script(js1)
        browser.execute_script(js2)
        browser.execute_script(js3)
        browser.execute_script(js4)


run = SDLCProcess()
run.Method3()

''' for x in range(5):
   print(runtestcase.test1()) '''