from selenium import webdriver
import time
import os
from selenium.webdriver import ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from Chrome.ValidationMethods import ValidationMethods


class FindElements_All:
    def test1(self):
        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)

        driver = webdriver.Chrome(driverlocation, chrome_options=opts)
        driver.get("http://automationpractice.com/index.php")
        driver.maximize_window()
        driver.implicitly_wait(5)
        elementCheck = ValidationMethods(driver)

        tab1 = driver.window_handles[0]
        driver.execute_script("window.open('https://www.phptravels.net/')")
        time.sleep(2)
        tab2 = driver.window_handles[1]
        time.sleep(2)
        driver.execute_script("window.open('http://the-internet.herokuapp.com/')")
        tab3 = driver.window_handles[2]
        driver.switch_to_window(tab2)
        time.sleep(2)
        #elem_by_xpath_1 = driver.find_element_by_xpath('//*[@id="content"]/ul/li[3]/a').click()

        css_1_HOTELS = '#search > div > div > div > div > div > nav > ul > li:nth-child(1) > a'
        elem_by_css_1_HOTELS = driver.find_element_by_css_selector(css_1_HOTELS)

        #Element Validation by isElementpresent Method
        elem_by_css_1_HOTELS_validation = elementCheck.isElementPresent(css_1_HOTELS, By.CSS_SELECTOR)
        print("BY_CSS--" + str(elem_by_css_1_HOTELS_validation) + str(driver.find_element_by_css_selector(css_1_HOTELS)))
        time.sleep(2)
        elem_by_css_1_HOTELS.click()

        XPATH_1_SEARCHTAB = '//span[@class="select2-chosen"]/a/span'
        #Validate Element BY elementList() method
        elem_by_XPATH_1_SEARCHTAB = driver.find_element_by_xpath(XPATH_1_SEARCHTAB)

        elem_by_XPATH_1_SEARCHTAB.click()
        time.sleep(2)
        elem_by_XPATH_1_SEARCHTAB.send_keys("Dhaka")

        css_2_SRCHLOC = '#select2-drop > ul > li > ul > li > div > span'
        elem_by_css_2_SRCHLOC = driver.find_element_by_css_selector(css_2_SRCHLOC)
        elem_by_css_2_SRCHLOC.click()

        XPATH_2_SRCHBTTN = '//*[@id="thhotels"]/form/div[5]/button'
        elem_by_XPATH_2_SRCHBTTN = driver.find_element_by_xpath(XPATH_2_SRCHBTTN )
        elem_by_XPATH_2_SRCHBTTN.click()

        css_3_CHCKIN = '#dpd1 > div > input'
        elem_by_css_3_CHCKIN = driver.find_element_by_css_selector(css_3_CHCKIN)

        css_3_CHCKIN_Month = 'body > div:nth-child(19) > div.datepicker-days > table > thead > tr:nth-child(1) > th.next'
        elem_by_css_3_CHCKIN_Month = driver.find_element_by_css_selector(css_3_CHCKIN_Month)

        i = elem_by_css_3_CHCKIN_Month.click()
        for i in range(3):
            print(elem_by_css_3_CHCKIN_Month.click())

        Xpath_3_CHCKIN_Date = '/html/body/div[13]/div[1]/table/tbody/tr[5]/td[3]'
        elem_by_Xpath_3_CHCKIN_Date = driver.find_element_by_xpath(Xpath_3_CHCKIN_Date)
        elem_by_Xpath_3_CHCKIN_Date.click()
        Xpath_3_CHCKout_RTNDate = '/html/body/div[14]/div[1]/table/tbody/tr[6]/td[6]'
        elem_by_Xpath_3_CHCKout_RTNDate = driver.find_element_by_xpath(Xpath_3_CHCKout_RTNDate)
        elem_by_Xpath_3_CHCKout_RTNDate.click()

        XPATH_4_Adults = '//*[@id="htravellersInput"]'
        elem_by_XPATH_4_Adults = driver.find_element_by_xpath(XPATH_4_Adults)
        elem_by_XPATH_4_Adults.click()
        ID_1_MINUSBTTN = 'hadultMinusBtn'
        elem_by_ID_1_MINUSBTTN = driver.find_element_by_id(ID_1_MINUSBTT)
        time.sleep(2)
        elem_by_ID_1_MINUSBTTN.click()
        elem_by_XPATH_2_SRCHBTTN.click()
        time.sleep(2)
        #driver.close()
        #body > div:nth-child(19) > div.datepicker-days > table > thead > tr:nth-child(1) > th.next
        #Next Step will be validation

    def test2(self):
        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        driver = webdriver.Chrome(driverlocation, chrome_options=opts)

        driver.get("http://automationpractice.com/index.php")

        elementCheck = ValidationMethods(driver)

        elem_by_id_validation = elementCheck.isElementPresent("search_query_top", By.ID)
        print("BY_Id--" + str(elem_by_id_validation) + str(driver.find_element_by_id("search_query_top")))

        elem_by_id = driver.find_element_by_id("search_query_top").click()
        time.sleep(1)

        elem_by_name_validations = elementCheck.elementList("submit_search", By.NAME)
        print("BY_Name--" + str(elem_by_name_validations))

        elem_by_name = driver.find_element_by_name("submit_search").click()
        time.sleep(2)
        driver.back()

        elem_by_class_validations = elementCheck.isElementPresent("sf-with-ul", By.CLASS_NAME)
        print("BY_ClassName--" + str(elem_by_class_validations))

        elem_by_class = driver.find_element_by_class_name("sf-with-ul").click()
        time.sleep(1)
        driver.back()

        elem_by_linktext_validations = elementCheck.isElementPresent("Contact us", By.LINK_TEXT)
        print("BY_linktext--" + str(elem_by_linktext_validations))

        elem_by_linktext1_validations = elementCheck.elementList("Contact us", By.LINK_TEXT)
        print("BY_linktext1--" + str(elem_by_linktext1_validations))

        elem_by_linktext = driver.find_element_by_link_text('Contact us').click()
        time.sleep(1)
        driver.back()

        # calling an output for the validation of the element
        elem_by_partial_link_validations = elementCheck.elementList("Sign", By.PARTIAL_LINK_TEXT)
        print("BY_partial_link--" + str(elem_by_partial_link_validations))


        # calling an output for the validation of the element
        elem_by_partial_link1_validations = elementCheck.isElementPresent("Sign", By.PARTIAL_LINK_TEXT)
        print("BY_partial_link1--" + str(elem_by_partial_link1_validations))

        elem_by_partial_link = driver.find_element_by_partial_link_text('Sign').click()
        time.sleep(1)

        elem_by_tag_validations = elementCheck.elementList("a", By.TAG_NAME)
        print("BY_tag--" + str(elem_by_tag_validations))

        elem_by_tag1_validations = elementCheck.isElementPresent("a", By.TAG_NAME)
        print("BY_tag1--" + str(elem_by_tag1_validations))

        elem_by_tag = driver.find_element_by_tag_name('a').click()
        time.sleep(1)

        elem_by_xpath_validations = elementCheck.elementList("(//*[@id='header']/div[3]/div/div/div[3]/div/a)", By.XPATH)
        print("BY_xpath--" + str(elem_by_xpath_validations))

        elem_by_xpath1_validations = elementCheck.isElementPresent("(//*[@id='header']/div[3]/div/div/div[3]/div/a)", By.XPATH)
        print("BY_xpath1--" + str(elem_by_xpath1_validations))

        elem_by_xpath = driver.find_element_by_xpath('//*[@id="header"]/div[3]/div/div/div[3]/div/a').click()
        time.sleep(1)


        #This is calling a function that require mouse movement to the element 
        women = driver.find_element_by_xpath('//*[@id="block_top_menu"]/ul/li[1]/a')

        hidden_submenu_tops = driver.find_element_by_xpath('//*[@id="block_top_menu"]/ul/li[1]/ul/li[1]/a')

        actions = ActionChains(driver)

        actions.move_to_element(women).pause(1).click(hidden_submenu_tops)

        elem_xpath2_actions_validations = elementCheck.isElementPresent("//*[@id='block_top_menu']/ul/li[1]/a", By.XPATH)

        print("BY_women_actions--" + str(elem_xpath2_actions_validations))

        elem_by_xpath3_actions_validations = elementCheck.elementList("//*[@id='block_top_menu']/ul/li[1]/a", By.XPATH)
        print("BY_women_actions--" + str(elem_by_xpath3_actions_validations))

        elem_by_xpath_actions_validations = elementCheck.isElementPresent("//*[@id='block_top_menu']/ul/li[1]/ul/li[1]/a", By.XPATH)
        print("BY_submenu_actions--" + str(elem_by_xpath_actions_validations))

        elem_by_xpath1_actions_validations = elementCheck.elementList("//*[@id='block_top_menu']/ul/li[1]/ul/li[1]/a", By.XPATH)
        print("BY_submenu_action --" + str(elem_by_xpath1_actions_validations))

        actions.perform()

        time.sleep(1)


runtestcase = FindElements_All()
runtestcase.test1()

runtestcase = FindElements_All()

x = runtestcase.test1()
''' for x in range(5):
   print(runtestcase.test1()) '''