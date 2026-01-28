# STM32F446RE Dual-Board Communication: I2C & UART

### Author: Md. Al-Amin Babu  
### Session: 2020 - 2021  
### Department of Computer Science and Engineering  
### University of Rajshahi, Bangladesh  

Welcome to the STM32F446RE Dual-Board Communication project! This repository demonstrates how to establish communication between two STM32F446RE Nucleo-64 boards using both the **UART** and **I2C** protocols, implemented through the **STM32 HAL library**.

---

## 📡 Communication Protocols

### 🟦 1. UART Communication (Point-to-Point)

The **UART (Universal Asynchronous Receiver-Transmitter)** protocol is employed here for asynchronous, full-duplex data exchange between two boards. In this setup:
- **USART1** is used for board-to-board communication.
- **USART2** is left available for PC debugging via the USB cable.

#### 🔌 Hardware Connection

| **Board A (TX/RX)** | **Board B (TX/RX)** | **Description** |
| -------------------- | -------------------- | --------------- |
| PA9 (TX)             | PA10 (RX)            | Transmit data from Board A to Board B |
| PA10 (RX)            | PA9 (TX)             | Receive data on Board A from Board B |
| GND                  | GND                  | Common Ground (Mandatory) |

#### ⚙️ Configuration (STM32CubeIDE)
- **Peripheral:** USART1
- **Mode:** Asynchronous
- **Baud Rate:** 115200 Bits/s
- **Word Length:** 8 Bits
- **Stop Bits:** 1
- **Parity:** None

#### 📝 Logic Implementation
- **Transmission:** Uses `HAL_UART_Transmit()` to send command strings.
- **Reception:** Uses `HAL_UART_Receive_IT()` (Interrupt Mode) to ensure the board can perform other tasks while waiting for data.

#### Workflow
When the **Blue User Button** is pressed on **Board A**, a message is sent to **Board B**, which then toggles an LED upon receiving the data.

---

### 🟧 2. I2C Communication (Master-Slave)

The **I2C (Inter-Integrated Circuit)** protocol is employed for synchronous communication. In this project:
- **Board A** acts as the **Master**.
- **Board B** acts as the **Slave** with a specific 7-bit hardware address.

#### 🔌 Hardware Connection

| **Board A (Master)** | **Board B (Slave)** | **Description** |
| --------------------- | -------------------- | --------------- |
| PB8 (SCL)             | PB8 (SCL)            | Serial Clock Line |
| PB9 (SDA)             | PB9 (SDA)            | Serial Data Line |
| GND                   | GND                  | Common Ground (Mandatory) |

> **Note:** For longer wires or high-speed communication, use **4.7kΩ resistors** externally to pull up the SCL and SDA lines.

#### ⚙️ Configuration (STM32CubeIDE)
- **Peripheral:** I2C1
- **I2C Speed:** Standard Mode (100kHz)
- **Slave Address (Board B):** 0x12
- **Addressing Mode:** 7-bit

#### 📝 Logic Implementation
- **Master (Board A):** Uses `HAL_I2C_Master_Transmit()` to send data and `HAL_I2C_Master_Receive()` to request data.
- **Slave (Board B):** Uses `HAL_I2C_Slave_Receive_IT()` to listen for its 7-bit address (0x12) on the bus.

#### Workflow
The **Master** (Board A) periodically requests sensor data or status updates from the **Slave** (Board B). If Board B recognizes the address, it sends back a data byte.

---

## 🛠 Project Structure

