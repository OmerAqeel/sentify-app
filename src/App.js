import logo from './logo.svg';
import './App.css';
import { Header } from './components/Header';
import { MainContentArea } from './components/MainContentArea';

function App() {
  return (
    <div className="App">
      <Header />
      <MainContentArea />
    </div>
  );
}

export default App;
