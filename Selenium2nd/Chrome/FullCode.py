from selenium import webdriver
import os
from selenium.webdriver import ChromeOptions



class BasicSetup:
    def test(self):

        driverlocation = "C:\\Users\\jzo_0\\PycharmProjects\\chromedriver.exe"
        os.environ["webdriver.chrome.driver"] = driverlocation
        opts = ChromeOptions()
        opts.add_experimental_option("detach", True)
        driver = webdriver.Chrome(driverlocation, chrome_options=opts)
        driver.get("http://automationpractice.com/index.php")



runrun = BasicSetup()
runrun.test()

''' Many a time a test fails to click on an element or enter text in a field as the element is
disabled or exists in the DOM, but is not displayed on the page; this will result in an error
being thrown and the test resulting in failures. For building reliable tests that can run
unattended, a robust exception and error handling is needed in the test flow.
We can handle these problems by checking the state of elements. The WebElement class
provides the following methods to check the state of an element:
Method Purpose
isEnabled() This method checks if an element is enabled. Returns true if enabled,
else false for disabled.
isSelected() This method checks if element is selected (radio button, checkbox, and
so on). It returns true if selected, else false for deselected
isDisplayed() This method checks if element is displayed.
In this recipe, we will use some of these methods to check the status and handle
possible errors. '''

...
• find_element_by_id()
• find_element_by_name()
• find_element_by_xpath()
• find_element_by_css_selector()
• find_element_by_link_text()
• find_element_by_partial_link_text()
• find_element_by_class_name()
• find_element_by_tag_name()\

Apart	from	the	methods	given	above,	there is	one	more method
which	might	be	useful	->	find_element().
driver.find_element(By.XPATH,		"xpath	expression")
These	are	the	attributes	available	for	By:
• ID	=	"id"
• NAME	=	"name"
• XPATH	=	"xpath	expression"
• CSS_SELECTOR	=	"css	selector	expression"
• LINK_TEXT	=	"link	text"
• PARTIAL_LINK_TEXT	=	"partial	link	text"
• CLASS_NAME	=	"class	name"
• TAG_NAME	=	"tag	name"

Finding	Multiple	Elements
• find_elements_by_name()
• find_elements_by_xpath()
• find_elements_by_link_text()
• find_elements_by_partial_link_text()
• find_elements_by_tag_name()
• find_elements_by_class_name()
• find_elements_by_css_selector()
driver.find_elements(By.XPATH,		"xpath	expression")
These	are	the	attributes	available	for	By:
• ID	=	"id"
• NAME	=	"name"
• XPATH	=	"xpath	expression"
• CSS_SELECTOR	=	"css	selector	expression"
• LINK_TEXT	=	"link	text"
• PARTIAL_LINK_TEXT	=	"partial	link	text"
• CLASS_NAME	=	"class	name"
• TAG_NAME	=	"tag	name"