from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
class ValidationMethods():

    def __init__(self, driver):
        self.driver = driver

    def isElementPresent(self, locator, byType):
      try:
        element = self.driver.find_element(byType, locator)
        if element is not None:
            print("Element is found")
            return True
        else:
            return False
      except:
             return False

    def elementList(self, locator, byType):
        try:
            element = self.driver.find_elements(byType, locator)
            if len(element) > 0:
                print("element is found" + str(element))
                return True
            else:
                return False
        except:
            return False

    def is_checked(self, driver, item):
             checked = driver.execute_script(("return document.getElementById('%s').checked") % item)
             return checked