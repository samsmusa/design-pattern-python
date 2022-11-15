from abc import ABC, abstractmethod


class Subject(ABC):
    """
    this is an abstract class
    """
    @abstractmethod
    def register_observer(self, observer):
        pass

    @abstractmethod
    def remove_observer(self, observer):
        pass

    @abstractmethod
    def notify_observers(self):
        pass


class Observer(ABC):
    """
    abstract class for observer
    """

    @abstractmethod
    def update(self, temp, humidity, pressure):
        pass


class DisplayElement:
    """
    abstract class for display
    """
    def display(self):
        raise NotImplementedError


class WeatherData(Subject, ABC):
    """
    implementation of subject interface
    """
    def __init__(self):
        self._temperature = None
        self._humidity = None
        self._pressure = None
        self._observers = []

    def register_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def notify_observers(self):
        for observer in self._observers:
            observer.update(self._temperature, self._humidity, self._pressure)

    def set_measurements(self, temperature: float, humidity:float, pressure: float):
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.notify_observers()


class CurrentConditionsDisplay(DisplayElement, Observer, ABC):
    def __init__(self, weather_data: WeatherData):
        self._temperature = None
        self._humidity = None
        self._weather_data = weather_data
        weather_data.register_observer(self)

    def display(self):
        print(
            f"current conditions: "
            f"{self._temperature:.1f}F degrees and"
            f"{self._humidity:.1f}% humidity"
        )

    def update(self, temp, humidity, pressure):
        self._temperature = temp
        self._humidity = humidity
        self.display()


class StatisticsDisplay(DisplayElement, Observer, ABC):
    def __init__(self, weather_data: WeatherData):
        self._max_temp = 0.
        self._min_temp = 200.
        self._temp_sum = 0.
        self._num_readings = 0
        self._weather_data = None
        self._weather_data = weather_data
        weather_data.register_observer(self)


