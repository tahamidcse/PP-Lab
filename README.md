# STM32F446RE Dual-Board Communication: I2C & UART (Register-Level)

| **Author**           | **Md. Al-Amin Babu**                         |
| -------------------- | -------------------------------------------- |
| **Session**          | 2020 - 2021                                  |
| **Department**       | Computer Science and Engineering             |
| **University**       | University of Rajshahi, Bangladesh          |

This repository demonstrates how to establish communication between two STM32F446RE Nucleo-64 boards using **UART** and **I2C** protocols with direct register-level configuration (no HAL library).

---

## 📡 Communication Protocols

### 🟦 1. UART Communication (Point-to-Point)

The **UART (Universal Asynchronous Receiver-Transmitter)** protocol is used for asynchronous, full-duplex data exchange between two boards. Here, **USART1** is used for board-to-board communication, while **USART2** is left for PC debugging via USB.

#### 🔌 Hardware Connection

| **Board A (TX/RX)** | **Board B (TX/RX)** | **Description** |
| -------------------- | -------------------- | --------------- |
| PA9 (TX)             | PA10 (RX)            | Transmit data from Board A to Board B |
| PA10 (RX)            | PA9 (TX)             | Receive data on Board A from Board B |
| GND                  | GND                  | Common Ground (Mandatory) |

#### ⚙️ Configuration (Register-Level)

- **Peripheral:** USART1
- **Mode:** Asynchronous
- **Baud Rate:** 115200 Bits/s
- **Word Length:** 8 Bits
- **Stop Bits:** 1
- **Parity:** None

#### 📝 Logic Implementation
- **Transmission:** Directly configure the **USART1** registers for data transmission using `USART1->DR`.
- **Reception:** Configure **USART1** for interrupt-driven data reception using `USART1->SR` to monitor the **RXNE** (Receive Not Empty) flag.

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

#### ⚙️ Configuration (Register-Level)

- **Peripheral:** I2C1
- **I2C Speed:** Standard Mode (100kHz)
- **Slave Address (Board B):** 0x12 (7-bit address)
- **Addressing Mode:** 7-bit

#### 📝 Logic Implementation

- **Master (Board A):** Configure **I2C1** registers for data transmission and reception by writing to `I2C1->DR` for transmission and monitoring the **TXE** flag for data readiness.
  
- **Slave (Board B):** Configure **I2C1** as a slave device and use the **I2C1->SR1** to detect the **ADDR** flag indicating address matching, then process the data accordingly.

#### Workflow
- The **Master** (Board A) periodically sends data or requests data from the **Slave** (Board B).
- The **Slave** (Board B) sends a data byte back if the address matches **0x12**.

---

## 🛠 Project Structure

