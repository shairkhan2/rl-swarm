import { AlchemyTransport, gensynTestnet } from "@account-kit/infra";
import { UserData } from "../db";
import { Address, toAccount } from "viem/accounts";
import { WalletClientSigner } from "@aa-sdk/core";
import { createWalletClient } from "viem";

export function createSignerForUser(
  user: UserData,
  transport: AlchemyTransport,
): WalletClientSigner {
  const signerAccount = toAccount({
    address: user.address as Address,
    signMessage: async () => {
      throw new Error("Not implemented");
    },
    signTransaction: async () => {
      throw new Error("Not implemented");
    },
    signTypedData: async () => {
      throw new Error("Not implemented");
    },
  });

  const walletClient = createWalletClient({
    account: signerAccount,
    chain: gensynTestnet,
    transport,
  });

  return new WalletClientSigner(walletClient, "custom");
}
