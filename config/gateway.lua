-- Edit addresses once. The controller's modem address must be written into
-- server/rack_agent.lua on every rack server during bootstrap.
return {
  install_root = "/openllm",
  coordinator = "REPLACE_WITH_SERVER_0_MODEM_ADDRESS",
  workers = { [1]="REPLACE_WITH_SERVER_1_MODEM_ADDRESS", [2]="REPLACE_WITH_SERVER_2_MODEM_ADDRESS", [3]="REPLACE_WITH_SERVER_3_MODEM_ADDRESS" },
  timeout = 20,
  release = "https://raw.githubusercontent.com/CoffeeSF/OpenLLM/main",
}
