export class ExperimentWebSocket {
    private ws: WebSocket | null = null;
    private messageHandlers: ((data: any) => void)[] = [];

    constructor(private url: string) {}

    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.messageHandlers.forEach(handler => handler(data));
        };

        this.ws.onclose = () => {
            setTimeout(() => this.connect(), 1000);
        };
    }

    onMessage(handler: (data: any) => void) {
        this.messageHandlers.push(handler);
    }

    sendCommand(command: string, params?: any) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command, params }));
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

export default ExperimentWebSocket;