from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
import os
class OPenKeyWordFrame:

    def Method1(self):
        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        #driverlocation = os.environ["webdriver.chrome.driver"]
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        browser = webdriver.Chrome(driverlocation, chrome_options=opts)
        js1 = 'window.open("https://www.browserstack.com/guide/keyword-driven-framework-in-selenium","_blank");'
        js2 =  'window.open("https://www.softwaretestinghelp.com/keyword-driven-framework-in-selenium/","_blank");'
        js3 =  'window.open("https://examples.javacodegeeks.com/enterprise-java/selenium/selenium-keyword-driven-framework-tutorial/","_blank");'
        js4 = 'window.open("https://www.h2kinfosys.com/blog/selenium-automation-framework-data-driven-keyword-driven-hybrid/","_blank");'
        js5 = 'window.open("https://www.guru99.com/creating-keyword-hybrid-frameworks-with-selenium.html","_blank");'
        browser.execute_script(js1)
        browser.execute_script(js2)
        browser.execute_script(js3)
        browser.execute_script(js4)
        browser.execute_script(js5)


        #driver.implicitly_wait(3)
        #ArrayList < String > tabs = new ArrayList < String > (driver.getWindowHandles());
        #driver.switchTo().window(tabs.get(1)); // secondtab
        #browser.execute_script('''window.open("http://example.com","_blank");''')

        #browser.execute_script('''window.open("http://example.com","_blank");''')

        #browser = webdriver.Chrome()
       # browser.get('http:/reddit.com')
        #window_before = driver.window_handles[0]
       # browser.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 't')
        #window_after = driver.window_handles[1]
        #driver.switch_to_window(window_after)
       # time.sleep(3)


        #from selenium import webdriver

       #second_page = "https://www.facebook.com/"
        #options = webdriver.ChromeOptions()
        #options.add_argument("start-maximized")
       # options.add_argument('disable-infobars')
        #driver = webdriver.Chrome(chrome_options=options, executable_path=r'C:\Utility\BrowserDrivers\chromedriver.exe')
       # driver.get(first_page)
       # driver.execute_script("window.open('" + second_page + "');")

run = OPenKeyWordFrame()
run.Method1()

''' for x in range(5):
   print(runtestcase.test1()) '''