import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from logger_all import logger_all
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logger_all.logger

@dataclass
class TimeframeMetrics:
    timeframe: str
    net_profit: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    market_condition: str
    trade_frequency: int

class CompilationError(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Strategy compilation failed: {errors}")

class EnhancedStrategyTester:
    TIMEFRAMES = ['1min']

    def __init__(self, tradingview_url: str, max_workers: int = 4):
        logging.info("EnhancedStrategyTester.__init__ called")
        self.tv_url = tradingview_url
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.drivers: Dict[str, webdriver.Firefox] = {}

    def get_or_create_driver(self, timeframe: str) -> webdriver.Firefox:
        logging.info(f"EnhancedStrategyTester.get_or_create_driver called with timeframe: {timeframe}")
        if timeframe not in self.drivers:
            profile_path = '/home/azoroth/.mozilla/firefox/gt4dwpe1.default-release'
            options = webdriver.FirefoxOptions()
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument(f'-profile {profile_path}')
            options.set_preference("dom.webdriver.enabled", False)
            options.set_preference('useAutomationExtension', False)
            driver = webdriver.Firefox(options=options)
            driver.set_page_load_timeout(90) # Increased page load timeout
            driver.set_window_size(1920, 1080)
            self.drivers[timeframe] = driver
            logger.info(f"Created new Firefox driver for timeframe: {timeframe}")
            return self.drivers[timeframe]
        return self.drivers[timeframe]

    async def test_strategy_timeframe(self, code: str, timeframe: str) -> TimeframeMetrics:
        logging.info(f"EnhancedStrategyTester.test_strategy_timeframe called with timeframe: {timeframe}")
        driver = self.get_or_create_driver(timeframe)
        try:
            logger.info(f"Starting strategy test for timeframe: {timeframe}")
            await asyncio.to_thread(driver.get, self.tv_url)
            logger.info("Successfully opened TradingView URL.")
            logger.info("Starting pasting code.")
            await self.paste_code(driver, code)
            logger.info("Code pasted successfully..")
            logger.info("Checking if strategy has been added to chart...")
            if not await self.check_strategy_added(driver):
                logger.warning(f"Strategy not added for timeframe {timeframe}. Checking for compilation errors...")
                # If there are compiler errors, handle them properly
                errors = await self.get_compiler_errors(driver)
                logger.error(f"Strategy compilation failed for timeframe {timeframe}: {errors}")
                raise CompilationError(errors) # Raise the CompilationError with the errors
            logger.info("Strategy added successfully..")
            logger.info("Proceeding with strategy tester...")
            # Proceed with testing the strategy if no errors were found
            await self.open_strategy_tester(driver)
            logger.info("Opened strategy tester.")
            logger.info("Selecting timeframe...")
            await self.select_timeframe(driver, timeframe)
            logger.info(f"Selected timeframe: {timeframe}.")
            logger.info("Fetching performance summary..")
            performance_data = await self.get_performance_summary(driver)
            logger.info(f"Performance summary fetched successfully... {performance_data}")
            logger.info("Fetching strategy properties...")
            properties_data = await self.get_strategy_properties(driver)
            logger.info(f"Strategy properties fetched.. {properties_data}")
            metrics = {
                'performance': performance_data,
                'properties': properties_data
            }
            tf_metrics = TimeframeMetrics(timeframe=timeframe, **metrics)
            logger.info(f"Completed strategy test for timeframe {timeframe}. Returning metrics.")
            return tf_metrics
        except CompilationError as ce:
            logger.error(f"Compilation error for timeframe {timeframe}: {ce.errors}")
            raise # Re-raise to stop further processing
        except Exception as e:
            logger.error(f"Unexpected error during strategy test for timeframe {timeframe}: {e}", exc_info=True)
            raise # Re-raise to stop further processing
        finally:
            await self.cleanup_driver(timeframe)
            logger.info(f"Cleaned up driver for timeframe {timeframe}.")

    async def test_strategy_all_timeframes(self, code: str):
        logging.info("EnhancedStrategyTester.test_strategy_all_timeframes called")
        results = []
        for tf in self.TIMEFRAMES:
            try:
                tf_metrics = await self.test_strategy_timeframe(code, tf)
                driver = self.get_or_create_driver(tf) # Driver is already created in test_strategy_timeframe
                performance_data = await self.get_performance_summary(driver)
                strategy_properties = await self.get_strategy_properties(driver)
                strategy_result = {
                    'timeframe': tf,
                    'metrics': tf_metrics,
                    'performance': performance_data,
                    'properties': strategy_properties
                }
                results.append(strategy_result)
                logger.info(f"Successfully tested and retrieved data for timeframe: {tf}")
                return True, results # Return success and results here when successful
            except CompilationError as ce:
                await self.cleanup_all_drivers() # Clean up on compilation error
                logger.info(f"Compilation error encountered: {ce.errors}") # Log compilation error
                return False, ce # Return False and the CompilationError object
            except Exception as e:
                logger.error(f"Error testing timeframe {tf}: {e}")
                continue # Continue to the next timeframe
        # await self.store_strategy_results(code, results) # Removed call to undefined method
        return True, results # Return success and results if all timeframes are successful

    async def cleanup_driver(self, timeframe: str):
        logging.info(f"EnhancedStrategyTester.cleanup_driver called with timeframe: {timeframe}")
        if timeframe in self.drivers:
            driver = self.drivers[timeframe]
            try:
                driver.quit() # Close the browser
                logger.info(f"Closed Firefox driver for timeframe: {timeframe}")
            except Exception as e:
                logger.error(f"Error closing driver for timeframe {timeframe}: {e}")
            finally:
                del self.drivers[timeframe] # Remove the driver from the dictionary

    async def cleanup_all_drivers(self):
        logging.info("EnhancedStrategyTester.cleanup_all_drivers called")
        for tf in list(self.drivers.keys()): # Iterate over a copy of the keys
            await self.cleanup_driver(tf)

    # async def store_strategy_results(self, code: str, results: List[Dict[str, Any]]): # commented out undefined method
    #     for result in results:
    #         timeframe = result['timeframe']
    #         tf_metrics = result['metrics'] # Access TimeframeMetrics object
    #         performance_data = result['performance']
    #         strategy_properties = result['properties']
    #         logger.info(f"Storing strategy results for timeframe: {timeframe}")
    #         logger.info(f"Timeframe Metrics: {tf_metrics}") # Log the TimeframeMetrics object
    #         logger.info(f"Performance Data: {performance_data}")
    #         logger.info(f"Strategy Properties: {strategy_properties}")

    async def get_performance_summary(self, driver: webdriver.Firefox) -> Dict[str, Any]:
        logging.info("EnhancedStrategyTester.get_performance_summary called")
        """Click on the Performance Summary tab and extract the relevant metrics."""
        try:
            # Click the 'Performance Summary' tab
            performance_summary_tab = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//button[@id='Performance Summary']"))
            )
            performance_summary_tab.click()
            # Wait for the Performance Summary to load and extract metrics
            performance_data = await self.extract_performance_data(driver)
            logger.info("Performance Summary data extracted successfully")
            return performance_data
        except Exception as e:
            logger.error(f"Error extracting performance summary: {e}")
            raise

    async def extract_performance_data(self, driver: webdriver.Firefox) -> Dict[str, float]:
        logging.info("EnhancedStrategyTester.extract_performance_data called")
        """Extract relevant performance data from the table."""
        performance_data = {}
        try:
            # Example of extracting a few key metrics from the Performance Summary
            performance_data['net_profit'] = await self._extract_metric(driver, 'Net Profit')
            performance_data['win_rate'] = await self._extract_metric(driver, 'Percent Profitable')
            performance_data['profit_factor'] = await self._extract_metric(driver, 'Profit Factor')
            performance_data['max_drawdown'] = await self._extract_metric(driver, 'Max Drawdown')
            performance_data['sharpe_ratio'] = await self._extract_metric(driver, 'Sharpe Ratio')
            performance_data['trade_frequency'] = await self._extract_metric(driver, 'Total Trades')
        except Exception as e:
            logger.error(f"Error extracting performance data: {e}")
            raise
        return performance_data

    async def get_strategy_properties(self, driver: webdriver.Firefox) -> Dict[str, Any]:
        logging.info("EnhancedStrategyTester.get_strategy_properties called")
        """Click on the Properties tab and extract the relevant strategy properties."""
        try:
            # Click the 'Properties' tab
            properties_tab = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//button[@id='Properties']"))
            )
            properties_tab.click()
            # Wait for the Properties section to load and extract relevant strategy properties
            properties_data = await self.extract_strategy_properties(driver)
            logger.info("Strategy Properties data extracted successfully")
            return properties_data
        except Exception as e:
            logger.error(f"Error extracting strategy properties: {e}")
            raise

    async def extract_strategy_properties(self, driver: webdriver.Firefox) -> Dict[str, Any]:
        logging.info("EnhancedStrategyTester.extract_strategy_properties called")
        """Extract strategy properties such as initial capital, order size, etc."""
        properties_data = {}
        try:
            # Example of extracting a few key strategy properties
            properties_data['initial_capital'] = await self._extract_property(driver, 'Initial capital')
            properties_data['order_size'] = await self._extract_property(driver, 'Order size')
            properties_data['pyramiding'] = await self._extract_property(driver, 'Pyramiding')
            properties_data['commission'] = await self._extract_property(driver, 'Commission')
            properties_data['slippage'] = await self._extract_property(driver, 'Slippage')
        except Exception as e:
            logger.error(f"Error extracting strategy properties: {e}")
            raise
        return properties_data

    async def _extract_metric(self, driver: webdriver.Firefox, metric_name: str) -> str:
        logging.info(f"EnhancedStrategyTester._extract_metric called with metric_name: {metric_name}")
        """Helper function to extract specific performance metrics."""
        try:
            element = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.presence_of_element_located((By.XPATH, f"//td[contains(text(), '{metric_name}')]/following-sibling::td"))
            )
            value = element.text.strip()
            return value
        except Exception as e:
            logger.error(f"Error extracting metric {metric_name}: {e}")
            return "N/A"

    async def _extract_property(self, driver: webdriver.Firefox, property_name: str) -> str:
        logging.info(f"EnhancedStrategyTester._extract_property called with property_name: {property_name}")
        """Helper function to extract specific property data."""
        try:
            element = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.presence_of_element_located((By.XPATH, f"//span[contains(text(), '{property_name}')]/following-sibling::span"))
            )
            value = element.text.strip()
            return value
        except Exception as e:
            logger.error(f"Error extracting property {property_name}: {e}")
            return "N/A"

    async def get_compiler_errors(self, driver):
        logging.info("EnhancedStrategyTester.get_compiler_errors called")
        try:
            logger.info("Starting to check for compiler errors...")
            error_elements = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.zone-widget'))
            )
            if not error_elements:
                logger.info("No compiler errors found.")
                return None

            dirname_elements = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.presence_of_all_elements_located((By.CLASS_NAME, 'dirname'))
            )
            if not dirname_elements: # Simplified check
                logger.error("No 'dirname' elements found.")
                return None

            dirname_text = dirname_elements[0].text.strip()
            logger.info(f"Found text in 'dirname' element: {dirname_text}")

            try: # More robust parsing
                num_errors = int(dirname_text.split(' of ')[1].split()[0])
                logger.info(f"Found {num_errors} errors.")
            except (IndexError, ValueError): # Handle parsing errors
                logger.error("Unable to parse the number of errors.")
                return None

            all_errors = await self.navigate_errors(driver, num_errors, start_at_last=True)
            if all_errors:
                return all_errors
            else:
                logger.info("No errors found after navigating.")
                return None
        except Exception as e:
            logger.error(f"Error gathering compiler errors: {e}", exc_info=True)
            return "Unknown error gathering logic" # Or handle differently

    async def navigate_errors(self, driver, num_errors, start_at_last=False):
        logging.info(f"EnhancedStrategyTester.navigate_errors called with num_errors: {num_errors}, start_at_last: {start_at_last}")
        try:
            logger.info('Navigating through errors...')
            actions = ActionChains(driver)
            all_errors = []

            if start_at_last:
                logger.info("Starting from the last error...")
                await asyncio.sleep(1)
                actions.key_down(Keys.ALT).send_keys(Keys.F8).key_up(Keys.ALT).perform()
                await asyncio.sleep(1)

            for i in range(num_errors):
                await asyncio.sleep(1) # Consistent sleep time
                if start_at_last and i > 0: # Combined conditions
                    actions.key_down(Keys.SHIFT).key_down(Keys.ALT).send_keys(Keys.F8).key_up(Keys.ALT).key_up(Keys.SHIFT).perform()
                else:
                    actions.key_down(Keys.ALT).send_keys(Keys.F8).key_up(Keys.ALT).perform()
                await asyncio.sleep(1)

                error_details = await self.extract_error_details(driver)
                all_errors.extend(error_details)

            logger.info(f"Successfully navigated through {num_errors} errors.")
            return all_errors
        except Exception as e:
            logger.error(f"Error navigating through errors: {e}", exc_info=True)
            return ["Error during navigation"]

    async def extract_error_details(self, driver):
        logging.info("EnhancedStrategyTester.extract_error_details called")
        try:
            error_messages = []
            error_elements = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, '.message-ADx-e-9'))
            )
            for error_element in error_elements:
                error_text = error_element.text.strip()
                error_messages.append(error_text)

            return error_messages
        except TimeoutException:
            logger.warning("Timeout waiting for error details. Proceeding without extracting specific error messages.")
            return ["Timeout occurred while extracting error details."]
        except Exception as e:
            logger.error(f"Error extracting error details: {e}", exc_info=True)
            return ["Error extracting error details."]

    async def paste_code(self, driver, code):
        logging.info("EnhancedStrategyTester.paste_code called")
        try:
            # Locate the Pine Editor
            pine_editor = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//div[@class=' InnerWrap-jWJyA1hM']//div[@class='view-line']"))
            )
            # Click to focus the editor
            pine_editor.click()
            actions = ActionChains(driver)
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).send_keys(Keys.DELETE).perform()
            actions.send_keys(code).perform()

            logger.info("Code pasted into Pine Editor.")
        except Exception as e:
            logger.error(f"Error pasting code: {e}", exc_info=True)
            raise

    async def check_strategy_added(self, driver):
        logging.info("EnhancedStrategyTester.check_strategy_added called")
        try:
            # Wait for the 'Add to Chart' button to be clickable
            add_to_chart_button = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Add to Chart']"))
            )

            # Click the 'Add to Chart' button
            add_to_chart_button.click()

            logger.info("'Add to Chart' button clicked.")
            return True
        except TimeoutException:
            logger.warning("Strategy not added to chart within the given timeout.")
            return False
        except Exception as e:
            logger.error(f"Error adding strategy to chart: {e}", exc_info=True)
            return False

    async def open_strategy_tester(self, driver):
        logging.info("EnhancedStrategyTester.open_strategy_tester called")
        try:
            # Click the three dots to open the menu
            three_dots_menu = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//div[@class='chart-page-header__menu-dots apply-common-tooltip']"))
            )
            three_dots_menu.click()

            # Click the 'Strategy Tester' option in the menu
            strategy_tester_option = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//div[text()='Strategy Tester']"))
            )
            strategy_tester_option.click()
            logger.info("Strategy Tester opened.")

        except Exception as e:
            logger.error(f"Error opening strategy tester: {e}", exc_info=True)
            raise

    async def select_timeframe(self, driver, timeframe):
        logging.info(f"EnhancedStrategyTester.select_timeframe called with timeframe: {timeframe}")
        try:
            # Click the timeframe selector
            timeframe_selector = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, "//div[@class='interval-picker-tv-dropdown-source']"))
            )
            timeframe_selector.click()

            # Click the specific timeframe option
            timeframe_option = WebDriverWait(driver, 60).until( # Increased WebDriverWait timeout
                EC.element_to_be_clickable((By.XPATH, f"//div[@class='item-Ne2dcHB6' and text()='{timeframe}']"))
            )
            timeframe_option.click()
            logger.info(f"Timeframe '{timeframe}' selected.")

        except Exception as e:
            logger.error(f"Error selecting timeframe '{timeframe}': {e}", exc_info=True)
            raise
