from selenium import webdriverfrom selenium.webdriver.common.by import Byimport osimport  timefrom selenium.webdriver import ChromeOptionsfrom selenium.webdriver.support.ui import WebDriverWaitfrom selenium.webdriver.support import expected_conditions as ECclass FindElementID:    def test1(self):        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"        os.environ["webdriver.chrome.driver"] = driverlocation        opts = ChromeOptions()        opts.add_experimental_option("detach", True)        driver = webdriver.Chrome(driverlocation, chrome_options=opts)        driver.get("http://automationpractice.com/index.php")        time.sleep(2)        element = driver.find_element_by_class_name("sf-with-ul").click()        driver.implicitly_wait(10)        time.sleep(2)        element2 = driver.find_element_by_xpath("//*[@id='center_column']/ul/li[1]/div/div[1]/div/a[1]/img").click()        wait = WebDriverWait(driver, 2)        element3 = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="thumb_2"]')))        driver.find_element_by_xpath("//*[@id='thumb_2']").click()        time.sleep(5)        driver.close()    hidden_submenu_tops = driver.find_element_by_xpath('//*[@id="block_top_menu"]/ul/li[1]/ul/li[1]/a')    def test2(self):        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"        os.environ["webdriver.chrome.driver"] = driverlocation        opts = ChromeOptions()        opts.add_experimental_option("detach", True)        driver = webdriver.Chrome(driverlocation, chrome_options=opts)        driver.get("http://automationpractice.com/index.php")runrun = FindElementID()runrun.test1()runrun = FindElementID()runrun.test2()